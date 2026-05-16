# app/main.py
#
# FastAPI application entry point.
#
# Responsibilities:
#   1. Load environment variables from .env (dev) or Render's env (production)
#   2. Configure application-wide logging
#   3. Register CORS middleware — required for Streamlit Cloud → Render cross-origin calls
#   4. Mount the API router
#   5. Expose the `app` object that uvicorn targets
#
# CORS NOTE:
#   allow_origins=["*"] is intentionally permissive for development.
#   Before going to production, replace "*" with your Streamlit Cloud URL:
#       allow_origins=["https://your-app-name.streamlit.app"]
#   This one change is all that's needed to lock down CORS for production.
#
# RUNNING LOCALLY:
#   uvicorn app.main:app --reload --port 8000
#
# RENDER DEPLOYMENT:
#   Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
#   (Render injects $PORT automatically)

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
# load_dotenv() reads .env in development.
# In production (Render), env vars are injected directly — load_dotenv()
# is a no-op when the variables are already in the environment, so this
# line is safe to leave in production code.
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan handler (startup / shutdown logging)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.
    The pipeline graph is already built at import time (see graph/pipeline.py),
    so this is purely for logging confirmation.
    """
    logger.info("PhysicsAgent API starting up...")
    yield
    logger.info("PhysicsAgent API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PhysicsAgent",
    description=(
        "A multi-agent system that answers physics questions using arXiv research papers. "
        "Built with LangGraph, LangChain, and gpt-4o-mini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# Required for Streamlit Cloud (different domain) to call this API.
# PRODUCTION: replace allow_origins=["*"] with your Streamlit Cloud URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← replace with specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(router)

logger.info("PhysicsAgent API ready. Docs at /docs")
