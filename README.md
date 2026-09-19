# PhysicsAgent 🔬

**A multi-agent system that answers physics questions using real arXiv research papers**, built with LangGraph and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** https://shahriar151-physicsagent.streamlit.app

---

## What It Does

Given a physics question, the system:

1. Breaks the question into focused arXiv search queries
2. Retrieves relevant research papers from arXiv
3. Synthesizes a cited answer grounded in those papers
4. Remembers conversation context for follow-up questions

**Example:**

Q: "How does quantum entanglement work?"
→ Retrieves relevant arXiv papers on entanglement dynamics
→ Returns an answer with inline citations and a reference list


---

## Tech Stack

Python · LangGraph · LangChain · FastAPI · Streamlit · arXiv API · OpenAI GPT-4o-mini

---

## Why I Built This

A project combining my Physics background with the programming and AI/ML tools I picked up through self-study — using language models to help search and synthesize physics literature from arXiv.

---

## Setup

```bash
git clone https://github.com/shahriar151/PhysicsAgent.git
cd PhysicsAgent
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

---

## Author

**Shahriar Mahmood**

---

## License

MIT License — see [LICENSE](LICENSE) for details.
