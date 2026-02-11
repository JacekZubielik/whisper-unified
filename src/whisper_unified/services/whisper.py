"""Embedded faster-whisper STT service."""

from typing import Any

import structlog
import torch
from faster_whisper import WhisperModel

from src.whisper_unified.config import Settings

logger = structlog.get_logger()


class EmbeddedWhisperService:
    """Embedded faster-whisper STT service (replaces external whisper container)."""

    def __init__(self, settings: Settings) -> None:
        self.model: WhisperModel | None = None
        self.model_name = settings.whisper_model_name
        self.compute_type = settings.whisper_compute_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "EmbeddedWhisperService initialized",
            model=self.model_name,
            compute_type=self.compute_type,
            device=self.device,
        )

    def load_model(self) -> None:
        """Load the Whisper model into memory."""
        logger.info("Loading Whisper model", model=self.model_name, device=self.device)
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio_path: str, language: str | None = None) -> dict[str, Any]:
        """Transcribe audio file using embedded faster-whisper."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        lang = language if language and language != "auto" else None

        segments, info = self.model.transcribe(
            audio_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )

        segment_list = []
        full_text_parts = []
        for segment in segments:
            segment_list.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
            )
            full_text_parts.append(segment.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segment_list,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
        }
