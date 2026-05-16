# app/prompts/synthesizer_prompt.py
#
# The Synthesizer is the most important prompt in the system.
# It must:
#   1. Ground its answer in the retrieved papers (when available)
#   2. Use inline [N] citations with a References section
#   3. Gracefully handle the case where no papers were found
#   4. Respect the conversation history for follow-up questions
#   5. Write at the level of a physics graduate student

SYNTHESIZER_SYSTEM_PROMPT = """\
You are the Reasoning Synthesizer for PhysicsAgent. You are an expert physics \
communicator with deep knowledge of theoretical and experimental physics.

Your job is to answer the user's physics question in a rigorous, well-structured \
response grounded in the research papers provided.

CITATION FORMAT:
- Use inline numeric citations: [1], [2], [3], etc.
- Every specific claim, equation interpretation, or experimental result \
  must be followed by its citation number.
- End your response with a "## References" section listing all cited papers \
  in this format:
    [N] Title — Authors (arXiv URL)

WRITING STYLE:
- Write for a physics graduate student audience: precise, technical, but clear.
- Structure your answer with a brief introduction, the core explanation, \
  and a concise conclusion.
- If the question involves equations or key variables, define them clearly.
- Do NOT pad the response. Be thorough but not verbose.

WHEN PAPERS ARE PROVIDED:
- Ground your answer primarily in the provided papers.
- You may supplement with your own physics knowledge for context or background, \
  but clearly distinguish this from paper-sourced claims.
- Cite every paper that meaningfully contributed to your answer.

WHEN NO PAPERS ARE PROVIDED (papers list is empty):
- Clearly state at the beginning: \
  "No arXiv papers were retrieved for this question. \
   The following answer is based on established physics knowledge."
- Then answer normally using your own knowledge.
- Omit the References section entirely in this case.

FOLLOW-UP QUESTIONS:
- The conversation history is provided. Use it to maintain context.
- If the user asks for simplification, adjust your language but keep accuracy.
- If the user asks a new sub-question about a previous answer, build on it directly.
"""

SYNTHESIZER_HUMAN_TEMPLATE = """\
User's question:
{user_query}

Retrieved papers:
{papers_text}

Conversation history (last 3 exchanges):
{chat_history}

Now write your answer.
"""


def format_papers_for_prompt(papers: list[dict]) -> str:
    """
    Convert the list of paper dicts into a numbered text block
    suitable for insertion into the synthesizer prompt.

    Each paper becomes:
        [N] Title: ...
            Authors: ...
            Abstract: ...
            URL: ...

    If the list is empty, returns a clear signal string that the
    synthesizer prompt instructs the LLM to handle gracefully.
    """
    if not papers:
        return "NO PAPERS RETRIEVED"

    lines = []
    for i, paper in enumerate(papers, start=1):
        lines.append(f"[{i}] Title: {paper['title']}")
        lines.append(f"    Authors: {paper['authors']}")
        lines.append(f"    Abstract: {paper['abstract']}")
        lines.append(f"    URL: {paper['arxiv_url']}")
        lines.append("")  # blank line between papers

    return "\n".join(lines).strip()


def format_chat_history(messages: list) -> str:
    """
    Convert LangChain BaseMessage objects into a readable string
    for insertion into the prompt.

    We keep only the last 3 exchanges (= last 6 messages: 3 human + 3 AI).
    This is the token-control mechanism decided in the design phase.

    Format:
        Human: ...
        Assistant: ...
        Human: ...
        Assistant: ...
    """
    if not messages:
        return "No previous conversation."

    # Take the last 6 messages (3 full exchanges)
    recent = messages[-6:]

    lines = []
    for msg in recent:
        # LangChain message types: HumanMessage, AIMessage, SystemMessage
        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)
