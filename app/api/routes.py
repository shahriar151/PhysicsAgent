# app/api/routes.py
#
# FastAPI route definitions for PhysicsAgent.
#
# Endpoints:
#   POST /solve  — main endpoint, accepts a physics question + thread_id
#   GET  /health — lightweight health check for Render's uptime monitoring
#
# THREAD_ID SEMANTICS:
#   The thread_id is the key MemorySaver uses to look up conversation history.
#   If the client sends the same thread_id on multiple requests, LangGraph
#   loads the previous state and the conversation continues.
#   If a new thread_id is sent, a fresh conversation starts.
#   The client (Streamlit frontend) is responsible for generating and
#   persisting the thread_id (e.g. using uuid4() on session start).
#
# ERROR HANDLING:
#   We catch broad exceptions and return a 500 with a clear message.
#   For a portfolio project, this is sufficient. In production you would
#   add more granular error types (arXiv timeout, OpenAI rate limit, etc.)

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.pipeline import physics_agent_graph

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="The physics question to answer.",
        examples=["How does quantum entanglement work?"],
    )
    thread_id: str = Field(
        ...,
        description="Unique conversation identifier. Reuse to continue a conversation.",
        examples=["user-session-abc123"],
    )


class SolveResponse(BaseModel):
    answer: str = Field(description="The cited answer from the Reasoning Synthesizer.")
    papers: list[dict] = Field(
        description="Papers used in this answer. Empty list if no papers were retrieved."
    )
    thread_id: str = Field(description="Echo of the thread_id for client convenience.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest) -> SolveResponse:
    """
    Accept a physics question and return a cited answer grounded in arXiv papers.

    The thread_id enables multi-turn conversations: send the same thread_id
    with a follow-up question and the system will maintain context from
    previous exchanges (last 3 exchanges are kept to control token usage).
    """
    logger.info(
        "POST /solve — thread_id='%s', question='%s'",
        request.thread_id,
        request.question[:80],  # Truncate long questions in logs
    )

    # LangGraph config: thread_id is passed here so MemorySaver can look up
    # (or initialise) the conversation checkpoint for this thread.
    config = {"configurable": {"thread_id": request.thread_id}}

    # Initial state for this invocation.
    # Only user_query needs to be set — all other fields have defaults.
    # LangGraph will merge this with the checkpointed state from previous turns.
    initial_state = {"user_query": request.question}

    try:
        # graph.invoke() runs the full pipeline synchronously and returns
        # the final state after all nodes have executed.
        final_state = physics_agent_graph.invoke(initial_state, config=config)

    except Exception as e:
        logger.error(
            "Pipeline error for thread_id='%s': %s",
            request.thread_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"PhysicsAgent pipeline error: {str(e)}",
        )

    return SolveResponse(
        answer=final_state["final_answer"],
        papers=final_state["papers"],
        thread_id=request.thread_id,
    )


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.
    Render uses this to verify the service is running.
    Returns 200 OK with a simple status payload.
    """
    return {"status": "ok", "service": "PhysicsAgent"}
