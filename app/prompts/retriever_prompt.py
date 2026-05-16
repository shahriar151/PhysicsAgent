# app/prompts/retriever_prompt.py
#
# The Paper Retriever does not call an LLM directly — it calls the
# search_arxiv tool once per sub-question and aggregates the results.
#
# This file exists for two reasons:
#   1. Consistency — every agent has a prompts file, making the codebase
#      easier to navigate and extend.
#   2. The retriever does log a status message so the graph's state
#      updates are traceable. That message template lives here.
#
# If you ever add an LLM-based re-ranking step to the retriever
# (e.g. "filter these 9 papers down to the 5 most relevant"),
# the system prompt for that step would go here.

# Used in paper_retriever.py to build a human-readable log of what was searched.
RETRIEVER_LOG_TEMPLATE = """\
Searched arXiv for {n_queries} sub-question(s).
Total papers retrieved: {n_papers}
Sub-questions searched:
{sub_questions_list}
"""
