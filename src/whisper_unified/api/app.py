"""FastAPI application factory."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.whisper_unified.api.routes import create_router
from src.whisper_unified.config import Settings
from src.whisper_unified.services.orchestrator import AudioOrchestrator

logger = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    orchestrator = AudioOrchestrator(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Load models on startup."""
        logger.info("Starting Whisper Unified service")

        orchestrator.whisper_service.load_model()

        success = await orchestrator.diarization_service.initialize()
        if not success:
            logger.error("Diarization service initialization failed")
            raise RuntimeError("Diarization service initialization failed")

        logger.info("All services initialized successfully")
        yield
        logger.info("Shutting down Whisper Unified")

    app = FastAPI(
        title="Whisper Unified",
        description="Embedded STT + Speaker Diarization + Redis Cache",
        version="3.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.orchestrator = orchestrator
    app.include_router(create_router())

    return app
