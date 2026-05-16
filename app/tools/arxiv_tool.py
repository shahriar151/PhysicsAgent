# app/tools/arxiv_tool.py
#
# This tool wraps the `arxiv` Python library as a LangChain @tool.
#
# DESIGN NOTE — Why not a live MCP server?
# The project context specifies MCP for the arXiv tool. We implement the
# interface exactly as an MCP tool would expose it (single string input,
# structured list output, typed docstring) so that swapping to a real MCP
# server later requires only changing the @tool decorator and import —
# the rest of the codebase stays identical.
#
# For the portfolio README: "arXiv retrieval is implemented via a LangChain
# tool with an MCP-compatible interface. Upgrading to a hosted MCP server
# requires only replacing the @tool decorator."

import arxiv
from langchain_core.tools import tool


@tool
def search_arxiv(query: str) -> list[dict]:
    """
    Search arXiv for physics research papers matching the given query.

    This tool is designed with an MCP-compatible interface: single string
    input, structured list output. It can be replaced with an MCP server
    call without changing any downstream agent code.

    Args:
        query: A focused search string (e.g. "quantum entanglement Bell inequality").
               Should be a single sub-question, not the full user question.

    Returns:
        A list of up to 3 paper dicts, each with keys:
            - title      (str): Full paper title
            - authors    (str): Comma-separated author names
            - abstract   (str): Full abstract text
            - arxiv_url  (str): Direct link to the paper on arxiv.org
        Returns an empty list if no results are found.
    """
    # arxiv.Search sorts by relevance by default, which is what we want.
    # max_results=3 matches our decision: 3 papers per sub-question.
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    # arxiv.Client() is the recommended way to run searches in arxiv >= 2.0
    client = arxiv.Client()

    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                # paper.authors is a list of arxiv.Author objects; .name gives the string
                "authors": ", ".join(author.name for author in paper.authors),
                "abstract": paper.summary,
                # entry_id is the canonical arXiv URL, e.g. https://arxiv.org/abs/2301.00001
                "arxiv_url": paper.entry_id,
            }
        )

    return results
