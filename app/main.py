"""
FastAPI Application – Salon AI Voice Agent.

Entry point for the server. Initializes all services on startup
and mounts the API routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.voice import router as voice_router
from app.services.rag_service import rag_service
from app.services.calendar_service import calendar_service
from app.services.voice_service import voice_service
from app.services.llm_agent import llm_agent
from app.logger.interaction_logger import interaction_logger

# ── Logging Setup ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("  🏪 Salon AI Voice Agent – Starting Up")
    logger.info("=" * 60)

    # Initialize services
    try:
        logger.info("Initializing RAG knowledge base...")
        rag_service.initialize()

        logger.info("Initializing interaction logger...")
        interaction_logger.initialize()

        logger.info("Initializing LLM agent...")
        llm_agent.initialize()

        logger.info("Initializing voice service...")
        voice_service.initialize()

        logger.info("Initializing calendar service...")
        try:
            calendar_service.initialize()
        except Exception as e:
            logger.warning(
                f"Calendar service init failed (expected if no credentials): {e}"
            )
            logger.warning("Calendar features will fail until credentials are configured.")

        logger.info("=" * 60)
        logger.info("  ✅ All services initialized – Ready for calls!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down Salon AI Voice Agent...")


# ── FastAPI App ───────────────────────────────────────
app = FastAPI(
    title="Salon AI Voice Agent",
    description=(
        "AI-powered voice agent for salon appointment management. "
        "Handles booking, rescheduling, cancellations, and customer inquiries "
        "via inbound phone calls."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(voice_router)


# ── Health & Info Endpoints ───────────────────────────
@app.get("/")
async def root():
    """Root endpoint – health check and info."""
    from app.services.call_orchestrator import call_orchestrator

    return {
        "service": "Salon AI Voice Agent",
        "status": "running",
        "active_calls": call_orchestrator.get_active_call_count(),
        "endpoints": {
            "incoming_call_webhook": "/voice/incoming",
            "media_stream_ws": "/voice/stream",
            "text_chat_test": "/voice/chat",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
