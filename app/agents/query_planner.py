# app/agents/query_planner.py
#
# Agent 1 — Query Planner
#
# Responsibility:
#   - Reads:  state.user_query, state.messages
#   - Writes: state.needs_search, state.sub_questions
#
# How it works:
#   1. Formats the conversation history and user query into the planner prompt.
#   2. Calls gpt-4o-mini and requests a strict JSON response.
#   3. Parses the JSON to extract needs_search and sub_questions.
#   4. Returns a partial state update (only the fields it owns).
#
# JSON parsing strategy:
#   We use response_format={"type": "json_object"} on the OpenAI call.
#   This is the most reliable way to get clean JSON from gpt-4o-mini —
#   it constrains the model at the API level, not just via prompt instructions.
#   No regex scraping needed.

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import PhysicsAgentState
from app.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_TEMPLATE
from app.prompts.synthesizer_prompt import format_chat_history

logger = logging.getLogger(__name__)


def run_query_planner(state: PhysicsAgentState) -> dict:
    """
    LangGraph node function for the Query Planner agent.

    LangGraph calls this function with the current state and expects a dict
    back containing only the fields this node wants to update. LangGraph
    merges this dict into the state — fields not in the returned dict are
    unchanged.

    Args:
        state: Current PhysicsAgentState (read-only in this function).

    Returns:
        A dict with keys: needs_search (bool), sub_questions (list[str])
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # Deterministic — we want consistent JSON structure
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    # Format conversation history for the prompt
    chat_history_text = format_chat_history(state.messages)

    # Build the message list for the LLM call
    human_content = PLANNER_HUMAN_TEMPLATE.format(
        chat_history=chat_history_text,
        user_query=state.user_query,
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    logger.info("Query Planner: calling LLM for query='%s'", state.user_query)

    response = llm.invoke(messages)

    # Parse the JSON response
    # response.content is a plain string because we used response_format=json_object
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as e:
        # If JSON parsing fails (shouldn't happen with json_object mode, but be safe),
        # default to running a search so the pipeline doesn't silently break.
        logger.error("Query Planner: JSON parse failed — %s. Raw: %s", e, response.content)
        return {
            "needs_search": True,
            "sub_questions": [state.user_query],  # Fall back to raw query
        }

    needs_search = bool(parsed.get("needs_search", True))
    sub_questions = parsed.get("sub_questions", [])

    # Defensive check: if needs_search is True but sub_questions is empty,
    # the Paper Retriever would have nothing to search. Fall back to raw query.
    if needs_search and not sub_questions:
        logger.warning(
            "Query Planner: needs_search=True but sub_questions is empty. "
            "Falling back to raw query as single sub-question."
        )
        sub_questions = [state.user_query]

    logger.info(
        "Query Planner: needs_search=%s, sub_questions=%s",
        needs_search,
        sub_questions,
    )

    result = {
        "needs_search": needs_search,
        "sub_questions": sub_questions,
    }

    # CRITICAL: When needs_search=False, LangGraph's invoke() with a fresh
    # initial_state dict can reset list fields to their defaults before the
    # checkpoint values are fully merged. We explicitly carry state.papers
    # forward here so the Synthesizer always has access to previously
    # retrieved papers on follow-up questions.
    if not needs_search and state.papers:
        result["papers"] = state.papers
        logger.info(
            "Query Planner: carrying forward %d existing papers for follow-up.",
            len(state.papers),
        )

    return result