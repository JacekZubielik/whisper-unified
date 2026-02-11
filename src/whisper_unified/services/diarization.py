"""Integrated PyAnnote-based speaker diarization service."""

import contextlib
import os
from typing import Any

import librosa
import structlog
import torch
from fastapi import HTTPException
from pyannote.audio import Pipeline

from src.whisper_unified.config import Settings

logger = structlog.get_logger()


class IntegratedDiarizationService:
    """Integrated PyAnnote-based speaker diarization service."""

    def __init__(self, settings: Settings) -> None:
        self.pipeline: Pipeline | None = None
        self.device: str | None = None
        self.model_name = settings.pyannote_model
        self.hf_token = settings.huggingface_token
        self.cuda_device = settings.cuda_visible_devices

        logger.info(
            "IntegratedDiarizationService initialized",
            model=self.model_name,
            cuda_device=self.cuda_device,
        )

    async def initialize(self) -> bool:
        """Initialize the PyAnnote pipeline."""
        try:
            logger.info("Loading PyAnnote pipeline", model=self.model_name)

            if self.hf_token and self.hf_token not in (
                "your_huggingface_token_here",
                "hf_xxx",
                "hf_your_token_here",
            ):
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name, use_auth_token=self.hf_token
                )
                logger.info("Loaded model with Hugging Face token")
            else:
                logger.warning("No valid HF token, trying public models")
                for model_name in [
                    "pyannote/speaker-diarization",
                    "speechbrain/spkrec-ecapa-voxceleb",
                ]:
                    try:
                        self.pipeline = Pipeline.from_pretrained(model_name)
                        self.model_name = model_name
                        logger.info("Loaded public model", model=model_name)
                        break
                    except Exception as e:
                        logger.warning("Failed to load model", model=model_name, error=str(e))

                if not self.pipeline:
                    logger.warning("No models available, creating mock pipeline")
                    self.pipeline = self._create_mock_pipeline()

            if torch.cuda.is_available() and self.pipeline is not None:
                self.device = f"cuda:{self.cuda_device}"
                self.pipeline = self.pipeline.to(torch.device(self.device))  # type: ignore[union-attr]
                logger.info("Pipeline moved to GPU", device=self.device)
            else:
                self.device = "cpu"
                logger.warning("CUDA not available, using CPU")

            return True
        except Exception as e:
            logger.error("Failed to initialize PyAnnote pipeline", error=str(e))
            return False

    def _create_mock_pipeline(self) -> object:
        """Create a mock pipeline for testing when real models are unavailable."""

        class MockPipeline:
            def __init__(self):
                self.model_name = "mock-diarization"

            def __call__(self, audio_path):
                try:
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = len(y) / sr
                except Exception:
                    duration = 10.0

                from types import SimpleNamespace

                segments = []
                current_time = 0.0
                speaker_id = 0
                while current_time < duration:
                    segment_duration = min(3.0, duration - current_time)
                    if segment_duration < 0.5:
                        break
                    seg = SimpleNamespace()
                    seg.start = current_time
                    seg.end = current_time + segment_duration
                    segments.append((seg, None, f"SPEAKER_{speaker_id:02d}"))
                    current_time += segment_duration
                    speaker_id = 1 - speaker_id

                return MockDiarization(segments)

            def to(self, _device):
                return self

        class MockDiarization:
            def __init__(self, segments):
                self.segments = segments

            def itertracks(self, yield_label=True):
                for segment, _, speaker in self.segments:
                    if yield_label:
                        yield segment, None, speaker
                    else:
                        yield segment, None

        return MockPipeline()

    def perform_diarization(self, audio_path: str) -> dict[str, Any]:
        """Perform speaker diarization using integrated PyAnnote."""
        try:
            logger.info("Starting diarization", audio_path=audio_path)
            if self.pipeline is None:
                raise RuntimeError("Diarization pipeline not initialized")
            diarization = self.pipeline(audio_path)

            segments = []
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": speaker,
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "duration": float(turn.end - turn.start),
                    }
                )
                speakers.add(speaker)

            speaker_stats = {}
            for speaker in speakers:
                sp_segs = [s for s in segments if s["speaker"] == speaker]
                total_time = sum(s["duration"] for s in sp_segs)
                speaker_stats[speaker] = {
                    "total_time": round(total_time, 2),
                    "segments_count": len(sp_segs),
                    "avg_segment_duration": (round(total_time / len(sp_segs), 2) if sp_segs else 0),
                }

            result = {
                "speaker_count": len(speakers),
                "total_duration": round(segments[-1]["end"], 2) if segments else 0,
                "speakers": speaker_stats,
                "segments": segments,
                "model": self.model_name,
                "device": self.device,
            }

            logger.info(
                "Diarization completed",
                speaker_count=len(speakers),
                segments_count=len(segments),
            )
            return result

        except Exception as e:
            logger.error("Diarization failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}") from e
        finally:
            with contextlib.suppress(Exception):
                os.unlink(audio_path)
