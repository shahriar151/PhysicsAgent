# app/agents/reasoning_synthesizer.py
#
# Agent 3 — Reasoning Synthesizer
#
# Responsibility:
#   - Reads:  state.user_query, state.papers, state.messages
#   - Writes: state.final_answer, appends to state.messages
#
# How it works:
#   1. Formats the papers list into numbered text (or a "NO PAPERS" signal).
#   2. Trims conversation history to the last 3 exchanges (6 messages).
#   3. Calls gpt-4o-mini with the synthesizer prompt.
#   4. Writes the answer to state.final_answer.
#   5. Appends BOTH the user's HumanMessage AND the AI's AIMessage to
#      state.messages so the next turn has full context.
#
# Why we append both messages here (not in the route handler):
#   The conversation history lives in LangGraph state and is persisted by
#   MemorySaver. The cleanest place to update it is inside the graph, not
#   outside. The route handler just reads state.final_answer.

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.state import PhysicsAgentState
from app.prompts.synthesizer_prompt import (
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_HUMAN_TEMPLATE,
    format_papers_for_prompt,
    format_chat_history,
)

logger = logging.getLogger(__name__)


def run_reasoning_synthesizer(state: PhysicsAgentState) -> dict:
    """
    LangGraph node function for the Reasoning Synthesizer agent.

    Produces the final cited answer and updates the conversation history.

    Args:
        state: Current PhysicsAgentState.

    Returns:
        A dict with keys:
            final_answer (str): The complete answer with citations.
            messages (list):    [HumanMessage(user_query), AIMessage(final_answer)]
                                LangGraph's add_messages reducer will APPEND these
                                to the existing messages list, not replace it.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,  # Slight creativity for natural prose, but still grounded
    )

    # Format inputs for the prompt
    papers_text = format_papers_for_prompt(state.papers)
    chat_history_text = format_chat_history(state.messages)

    logger.info(
        "Reasoning Synthesizer: synthesizing answer for query='%s' with %d papers",
        state.user_query,
        len(state.papers),
    )

    # Build the message list
    human_content = SYNTHESIZER_HUMAN_TEMPLATE.format(
        user_query=state.user_query,
        papers_text=papers_text,
        chat_history=chat_history_text,
    )

    llm_messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(llm_messages)
    final_answer = response.content

    logger.info("Reasoning Synthesizer: answer generated (%d chars)", len(final_answer))

    # Append to conversation history.
    # IMPORTANT: We return these as a list for LangGraph's add_messages reducer.
    # The reducer APPENDS items in this list to state.messages — it does NOT replace.
    # This is what makes multi-turn work: each call adds 2 messages,
    # and format_chat_history() trims to the last 6 before each LLM call.
    new_messages = [
        HumanMessage(content=state.user_query),
        AIMessage(content=final_answer),
    ]

    return {
        "final_answer": final_answer,
        "messages": new_messages,
    }
