# app/graph/pipeline.py
#
# GRAPH STRUCTURE:
#
#   START
#     │
#     ▼
#   query_planner
#     │
#     ├─── needs_search=True ───► paper_retriever ──► reasoning_synthesizer ──► END
#     │
#     └─── needs_search=False ──► passthrough_papers ──► reasoning_synthesizer ──► END
#
# WHY passthrough_papers EXISTS:
#   LangGraph invokes the graph with a fresh initial_state dict on every /solve
#   call. This dict only contains user_query. LangGraph merges the checkpoint
#   state AFTER applying the initial_state, but the exact merge timing means
#   that list fields with default_factory=list (like `papers`) can arrive at
#   nodes as [] even when the checkpoint has real data.
#
#   passthrough_papers is a minimal node that runs AFTER the Query Planner
#   has finished and AFTER LangGraph has fully merged the checkpoint. At that
#   point, state.papers is correctly populated from the checkpoint. The node
#   simply re-emits them so the Synthesizer is guaranteed to receive them.
#
# MEMORY:
#   MemorySaver persists PhysicsAgentState keyed by thread_id.
#   Future upgrade (one line): replace MemorySaver() with
#   AsyncPostgresSaver.from_conn_string(DATABASE_URL).

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import PhysicsAgentState
from app.agents.query_planner import run_query_planner
from app.agents.paper_retriever import run_paper_retriever
from app.agents.reasoning_synthesizer import run_reasoning_synthesizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Passthrough node (needs_search=False path only)
# ---------------------------------------------------------------------------

def passthrough_papers(state: PhysicsAgentState) -> dict:
    """
    Passthrough node that explicitly re-emits state.papers on the
    needs_search=False path so the Synthesizer always receives them.

    By the time this node runs, LangGraph has fully merged the checkpoint
    into the active state, so state.papers correctly holds the papers from
    the previous turn. Re-emitting them as a return value ensures they are
    present in the state snapshot the Synthesizer receives.
    """
    papers = state.papers or []
    logger.info(
        "Passthrough: forwarding %d papers to Synthesizer (no new search).",
        len(papers),
    )
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def should_search(state: PhysicsAgentState) -> str:
    """
    Routes to paper_retriever (new question) or passthrough_papers (follow-up).
    """
    if state.needs_search:
        logger.info("Router: needs_search=True → paper_retriever")
        return "paper_retriever"
    else:
        logger.info("Router: needs_search=False → passthrough_papers")
        return "passthrough_papers"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PhysicsAgentState)

    # Register all nodes
    graph.add_node("query_planner", run_query_planner)
    graph.add_node("paper_retriever", run_paper_retriever)
    graph.add_node("passthrough_papers", passthrough_papers)
    graph.add_node("reasoning_synthesizer", run_reasoning_synthesizer)

    # Entry point
    graph.add_edge(START, "query_planner")

    # Conditional routing after Query Planner
    graph.add_conditional_edges(
        "query_planner",
        should_search,
        {
            "paper_retriever": "paper_retriever",
            "passthrough_papers": "passthrough_papers",
        },
    )

    # Both paths converge at the Synthesizer
    graph.add_edge("paper_retriever", "reasoning_synthesizer")
    graph.add_edge("passthrough_papers", "reasoning_synthesizer")

    # Terminal node
    graph.add_edge("reasoning_synthesizer", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info("PhysicsAgent pipeline compiled successfully.")
    return compiled


# Module-level singleton — built once at import time
physics_agent_graph = build_graph()