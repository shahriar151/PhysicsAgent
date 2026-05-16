# frontend/streamlit_app.py
#
# PhysicsAgent — Streamlit Chat Frontend
#
# UI design: dark scientific aesthetic — deep navy background, monospace accents,
# clean typography. Feels like a research terminal, not a generic chatbot.
#
# Features:
#   - Chat interface with full conversation history displayed
#   - Papers retrieved shown in an expandable sidebar panel
#   - Thread ID persisted in session state (new ID on page load)
#   - Backend URL configurable via st.secrets or environment variable
#   - Clear conversation button resets thread_id and history

import uuid
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhysicsAgent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark scientific terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root theme */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: #1a2035;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #475569;
        --border: #1e293b;
        --user-bubble: #1e3a5f;
        --ai-bubble: #1a2035;
    }

    /* Override Streamlit defaults */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background-color: var(--bg-primary);
        border-bottom: 1px solid var(--border);
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 860px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Title */
    .app-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }

    .app-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
    }

    /* Chat message — user */
    .msg-user {
        background: var(--user-bubble);
        border: 1px solid #2563eb33;
        border-radius: 12px 12px 4px 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-left: 10%;
    }

    .msg-label-user {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: var(--accent-blue);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        text-align: right;
    }

    /* Chat message — assistant */
    .msg-ai {
        background: var(--ai-bubble);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-cyan);
        border-radius: 4px 12px 12px 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        color: var(--text-primary);
        font-size: 0.95rem;
        line-height: 1.7;
        margin-right: 5%;
    }

    .msg-label-ai {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: var(--accent-cyan);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* Input area */
    .stTextArea textarea {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px #06b6d420 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #0891b2);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    /* Paper card in sidebar */
    .paper-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
    }

    .paper-title {
        color: var(--accent-cyan);
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
        line-height: 1.4;
    }

    .paper-authors {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
        font-style: italic;
    }

    .paper-link {
        color: var(--accent-blue);
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-decoration: none;
    }

    /* Status / info chips */
    .status-chip {
        display: inline-block;
        background: #0f172a;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    /* Divider */
    hr {
        border-color: var(--border) !important;
        margin: 1rem 0 !important;
    }

    /* Spinner text */
    .stSpinner > div {
        color: var(--accent-cyan) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": "user"|"assistant", "content": str}

if "papers" not in st.session_state:
    st.session_state.papers = []  # Most recent papers retrieved

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------
# In production (Streamlit Cloud), set BACKEND_URL in Streamlit secrets:
#   [secrets]
#   BACKEND_URL = "https://your-app.onrender.com"
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Sidebar — papers + session info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="app-title">⚛ PhysicsAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">arXiv-grounded physics Q&A</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Session info
    st.markdown(
        f'<div class="status-chip">session: {st.session_state.thread_id}</div>',
        unsafe_allow_html=True,
    )

    # Clear conversation button
    if st.button("↺  New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.papers = []
        st.rerun()

    st.markdown("---")

    # Papers panel
    if st.session_state.papers:
        st.markdown(
            f'<span style="font-family: Space Mono, monospace; font-size: 0.72rem; '
            f'color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase;">'
            f'📄 {len(st.session_state.papers)} papers retrieved</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for i, paper in enumerate(st.session_state.papers, 1):
            # Truncate long abstracts
            abstract = paper.get("abstract", "")
            abstract_short = abstract[:200] + "..." if len(abstract) > 200 else abstract

            with st.expander(f"[{i}] {paper.get('title', 'Unknown')[:55]}..."):
                st.markdown(
                    f'<div class="paper-authors">{paper.get("authors", "")[:80]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color: #94a3b8; font-size: 0.8rem; '
                    f'line-height: 1.5; margin-bottom: 0.5rem;">{abstract_short}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<a class="paper-link" href="{paper.get("arxiv_url", "#")}" '
                    f'target="_blank">→ View on arXiv</a>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="color: #475569; font-size: 0.82rem; '
            'font-family: Space Mono, monospace;">No papers yet.<br>'
            'Ask a physics question.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="color: #334155; font-size: 0.72rem; font-family: Space Mono, monospace;">'
        'Powered by LangGraph + GPT-4o-mini<br>arXiv paper retrieval</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">PhysicsAgent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask any physics question — answers grounded in arXiv research papers</div>',
    unsafe_allow_html=True,
)

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-label-user">You</div>'
            f'<div class="msg-user">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Label in styled HTML
        st.markdown(
            '<div class="msg-label-ai">⚛ PhysicsAgent</div>',
            unsafe_allow_html=True,
        )
        # Answer uses st.markdown directly so that bold, headers,
        # inline citations [1][2], LaTeX, and References section
        # all render correctly. Wrapped in a styled container div.
        st.markdown(
            '<div class="msg-ai">',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown("<br>", unsafe_allow_html=True)
question = st.text_area(
    label="Your question",
    placeholder="e.g. How does Hawking radiation work? What is quantum decoherence?",
    height=90,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 6])
with col1:
    submit = st.button("Ask →")

# ---------------------------------------------------------------------------
# Handle submission
# ---------------------------------------------------------------------------
if submit and question.strip():
    # Add user message to history immediately
    st.session_state.messages.append({"role": "user", "content": question.strip()})

    with st.spinner("Searching arXiv and reasoning over papers..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/solve",
                json={
                    "question": question.strip(),
                    "thread_id": st.session_state.thread_id,
                },
                timeout=180,  # arXiv (slow API) + 3 sub-questions + LLM = up to 90s
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("answer", "No answer returned.")
            papers = data.get("papers", [])

            # Update session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.papers = papers

        except requests.exceptions.ConnectionError:
            error_msg = (
                "⚠️ Cannot connect to the PhysicsAgent backend. "
                "Make sure the FastAPI server is running at: `" + BACKEND_URL + "`"
            )
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except requests.exceptions.Timeout:
            error_msg = "⚠️ Request timed out. The arXiv search may be slow — please try again."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            error_msg = f"⚠️ Unexpected error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()

elif submit and not question.strip():
    st.warning("Please enter a question before submitting.")