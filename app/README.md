# PhysicsAgent 🔬

**A multi-agent AI system that answers physics questions using real arXiv research papers.**

Built with LangGraph, LangChain, FastAPI, and GPT-4o-mini. Supports multi-turn conversations with memory.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.5-blue?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo

| Service | URL |
|---|---|
| Streamlit Frontend | https://shahriar151-physicsagent.streamlit.app |
| FastAPI Docs | https://physicsagent-api-7jdn.onrender.com/docs |

---

## What It Does

A user asks a physics question. The system:

1. **Decomposes** the question into 2–4 focused arXiv search queries
2. **Retrieves** up to 3 relevant research papers per query from arXiv
3. **Synthesizes** a rigorous, cited answer grounded in those papers
4. **Remembers** the conversation — follow-up questions reuse retrieved papers without redundant API calls

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Query Planner  │  GPT-4o-mini — decomposes question into arXiv sub-queries
│   (Agent 1)     │  Detects follow-ups → skips retrieval if papers already exist
└────────┬────────┘
         │
    needs_search?
         │
    ┌────┴─────┐
   YES         NO
    │           │
    ▼           ▼
┌──────────┐ ┌──────────────────┐
│  Paper   │ │ Passthrough Node │  Carries existing papers forward
│Retriever │ │                  │  (no redundant arXiv calls)
│(Agent 2) │ └────────┬─────────┘
└────┬─────┘          │
     │                │
     └────────┬───────┘
              │
              ▼
┌─────────────────────┐
│ Reasoning           │  GPT-4o-mini — writes cited answer
│ Synthesizer         │  Inline [N] citations + References section
│   (Agent 3)         │  Reads last 3 exchanges from conversation memory
└─────────────────────┘
              │
              ▼
        Final Answer
   (with arXiv citations)
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph | Explicit state machine, conditional edges, built-in checkpointing |
| LLM | GPT-4o-mini | Cost-efficient, strong instruction following, JSON mode support |
| Memory | LangGraph MemorySaver | In-memory checkpointing keyed by `thread_id`; swap to PostgreSQL in one line |
| arXiv tool | `arxiv` Python library | MCP-compatible interface; designed for easy MCP server upgrade |
| API | FastAPI | Async, auto-docs, Pydantic validation, Render-compatible |
| Frontend | Streamlit | Rapid iteration, Streamlit Cloud deployment |

---

## Project Structure

```
physicsagent/
├── app/
│   ├── main.py                        # FastAPI app, CORS, lifespan
│   ├── api/
│   │   └── routes.py                  # POST /solve, GET /health
│   ├── agents/
│   │   ├── state.py                   # Shared LangGraph state (PhysicsAgentState)
│   │   ├── query_planner.py           # Agent 1 — query decomposition + routing
│   │   ├── paper_retriever.py         # Agent 2 — arXiv search + deduplication
│   │   └── reasoning_synthesizer.py   # Agent 3 — cited answer generation
│   ├── graph/
│   │   └── pipeline.py                # LangGraph StateGraph + MemorySaver
│   ├── tools/
│   │   └── arxiv_tool.py              # @tool wrapper — MCP-compatible interface
│   └── prompts/
│       ├── planner_prompt.py
│       ├── retriever_prompt.py
│       └── synthesizer_prompt.py
├── frontend/
│   └── streamlit_app.py               # Streamlit chat UI
├── .env.example
├── .gitignore
├── requirements.txt                   # Backend dependencies
├── requirements-frontend.txt          # Frontend dependencies
└── render.yaml                        # Render deployment config
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Conda (recommended) or venv
- OpenAI API key with GPT-4o-mini access

### 1. Clone the repository

```bash
git clone https://github.com/shahriar151/PhysicsAgent.git
cd PhysicsAgent
```

### 2. Create and activate environment

```bash
conda create -n physicsagent python=3.11 -y
conda activate physicsagent
conda install -c conda-forge libffi -y
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Run the backend

```powershell
# Windows PowerShell
$env:PYTHONPATH = "."
uvicorn app.main:app --reload
```

```bash
# Linux / macOS
PYTHONPATH=. uvicorn app.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

### 6. Run the frontend (separate terminal)

```bash
pip install -r requirements-frontend.txt
streamlit run frontend/streamlit_app.py
```

---

## API Reference

### `POST /solve`

Ask a physics question and receive a cited answer.

**Request:**
```json
{
  "question": "How does quantum entanglement work?",
  "thread_id": "my-session-001"
}
```

**Response:**
```json
{
  "answer": "Quantum entanglement is... [1][2]\n\n## References\n[1] Title — Authors (URL)",
  "papers": [
    {
      "title": "Entanglement dynamics in hybrid quantum circuits",
      "authors": "Andrew C. Potter, Romain Vasseur",
      "abstract": "...",
      "arxiv_url": "http://arxiv.org/abs/2111.08018v2"
    }
  ],
  "thread_id": "my-session-001"
}
```

**Multi-turn example:**
```bash
# Turn 1 — retrieves papers from arXiv
POST /solve  {"question": "Explain Hawking radiation", "thread_id": "t1"}

# Turn 2 — reuses papers, no arXiv call
POST /solve  {"question": "Can you simplify that?", "thread_id": "t1"}
```

### `GET /health`

```json
{"status": "ok", "service": "PhysicsAgent"}
```

---

## Tech Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Multi-agent orchestration with conditional edges and memory
- **[LangChain](https://python.langchain.com/)** — LLM abstraction and tool interface
- **[FastAPI](https://fastapi.tiangolo.com/)** — Async REST API with auto-generated docs
- **[Streamlit](https://streamlit.io/)** — Interactive web frontend
- **[arXiv Python](https://github.com/lukasschwab/arxiv.py)** — Research paper retrieval
- **[OpenAI GPT-4o-mini](https://platform.openai.com/docs/models)** — Language model backbone
- **[Pydantic v2](https://docs.pydantic.dev/)** — Data validation and state schema

---

## Future Improvements

- Replace MemorySaver with `AsyncPostgresSaver` for persistent cross-session memory
- Upgrade arXiv tool to a hosted MCP server for real-time paper access
- Add semantic reranking of retrieved papers before synthesis
- Stream the synthesizer response token-by-token to the frontend

---

## Author

**Shahriar** — Physics graduate, AI Engineer in training.
Preparing for MSc Applied AI (September 2026 intake).

[GitHub](https://github.com/shahriar151) · [LinkedIn](https://linkedin.com/in/shahriar151)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
