"""Entry point for `python -m src.whisper_unified`."""

import logging

import structlog
import uvicorn

from src.whisper_unified.api.app import create_app
from src.whisper_unified.config import Settings

logger = structlog.get_logger()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    logger.info("Starting Whisper Unified", host=settings.host, port=settings.port)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, reload=False, workers=1)


if __name__ == "__main__":
    main()
