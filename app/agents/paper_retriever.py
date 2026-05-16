# app/agents/paper_retriever.py
#
# Agent 2 — Paper Retriever
#
# PERFORMANCE FIX: Sub-questions are now searched CONCURRENTLY using
# ThreadPoolExecutor. Previously: 3 queries × ~15s each = ~45s sequential.
# Now: all 3 queries run in parallel = ~15s total. This is the primary
# fix for the Streamlit timeout issue.
#
# arXiv's API is HTTP-based and I/O bound — ThreadPoolExecutor is the
# correct tool here (not asyncio, which would require async refactoring
# of the entire LangGraph pipeline).

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.state import PhysicsAgentState
from app.tools.arxiv_tool import search_arxiv
from app.prompts.retriever_prompt import RETRIEVER_LOG_TEMPLATE

logger = logging.getLogger(__name__)


def _search_single(query: str) -> list[dict]:
    """
    Search arXiv for a single query. Designed to run in a thread pool.
    Returns an empty list on any error so one failed query doesn't
    kill the whole pipeline.
    """
    try:
        logger.info("Paper Retriever: searching arXiv for '%s'", query)
        return search_arxiv.invoke({"query": query})
    except Exception as e:
        logger.error("Paper Retriever: search_arxiv failed for '%s' — %s", query, e)
        return []


def run_paper_retriever(state: PhysicsAgentState) -> dict:
    """
    LangGraph node function for the Paper Retriever agent.

    Searches arXiv concurrently for all sub-questions and deduplicates results.

    Args:
        state: Current PhysicsAgentState.

    Returns:
        A dict with key: papers (list[dict])
        Each dict has: title, authors, abstract, arxiv_url
    """
    if not state.sub_questions:
        logger.warning("Paper Retriever: called with empty sub_questions.")
        return {"papers": []}

    all_papers = []
    seen_urls = set()

    # Run all sub-question searches concurrently.
    # max_workers=4 is safe — arXiv rate-limits by IP, not by connection count.
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_query = {
            executor.submit(_search_single, query): query
            for query in state.sub_questions
        }
        for future in as_completed(future_to_query):
            papers = future.result()
            for paper in papers:
                url = paper.get("arxiv_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_papers.append(paper)

    sub_questions_formatted = "\n".join(f"  - {q}" for q in state.sub_questions)
    logger.info(
        RETRIEVER_LOG_TEMPLATE.format(
            n_queries=len(state.sub_questions),
            n_papers=len(all_papers),
            sub_questions_list=sub_questions_formatted,
        )
    )

    # DIAGNOSTIC LOG — tells us exactly how many papers reach the Synthesizer
    logger.info(
        "Paper Retriever: returning %d deduplicated papers to state.",
        len(all_papers),
    )

    return {"papers": all_papers}