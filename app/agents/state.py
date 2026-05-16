# app/agents/state.py
#
# This is the single source of truth for all data flowing through the graph.
# Every agent reads from and writes to this state object.
# LangGraph's MemorySaver checkpoints this state keyed by thread_id,
# which is what gives us multi-turn conversation support.

from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PhysicsAgentState(BaseModel):
    """
    Shared state passed between all three agents in the PhysicsAgent pipeline.

    Field-by-field ownership:
        user_query     — set by the API route before the graph runs
        needs_search   — set by Query Planner (True = fetch new papers, False = reuse existing)
        sub_questions  — set by Query Planner (only when needs_search=True)
        papers         — set by Paper Retriever (carries over across turns when needs_search=False)
        final_answer   — set by Reasoning Synthesizer
        messages       — append-only conversation history, managed by LangGraph's add_messages
                         reducer. Keeps a rolling window of the last 3 exchanges (trimmed
                         inside the Synthesizer before the LLM call to control token usage).
    """

    user_query: str = ""

    # Query Planner sets this flag to tell the graph whether to run the
    # Paper Retriever or skip straight to the Reasoning Synthesizer.
    needs_search: bool = True

    sub_questions: list[str] = Field(default_factory=list)

    # Each paper dict has exactly 4 keys:
    #   { "title": str, "authors": str, "abstract": str, "arxiv_url": str }
    # Keeping the schema flat (not a nested Pydantic model) makes it easy
    # to serialize directly into the FastAPI response and into prompt text.
    papers: list[dict] = Field(default_factory=list)

    final_answer: str = ""

    # Annotated with add_messages so LangGraph knows to APPEND new messages
    # rather than overwrite the list on each state update.
    # This is what enables multi-turn memory across /solve calls on the same thread_id.
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
