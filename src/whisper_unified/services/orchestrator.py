"""Central audio processing orchestrator — coordinates STT, diarization, cache, uploads."""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import redis.asyncio as aioredis
import soundfile as sf
import structlog
from fastapi import HTTPException, UploadFile
from langdetect import DetectorFactory, detect

from src.whisper_unified.config import Settings
from src.whisper_unified.services.diarization import IntegratedDiarizationService
from src.whisper_unified.services.whisper import EmbeddedWhisperService

# Configure deterministic language detection
DetectorFactory.seed = 0

logger = structlog.get_logger()


class AudioOrchestrator:
    """Coordinates STT, diarization, cache, and file upload management."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis_url = settings.redis_url
        self.enable_caching = settings.enable_caching
        self.cache_ttl = settings.cache_ttl
        self.language_detection = settings.language_detection
        self.speaker_diarization = settings.speaker_diarization

        # Embedded services
        self.whisper_service = EmbeddedWhisperService(settings)
        self.diarization_service = IntegratedDiarizationService(settings)

        # Auto-transcription control
        self.auto_transcription = settings.whisper_auto_transcription
        self.upload_only_mode = settings.whisper_upload_only_mode
        self.require_explicit_start = settings.whisper_require_explicit_start
        self.show_file_info = settings.whisper_show_file_info_on_upload

        # Default parameters
        self.default_language = settings.whisper_default_language
        self.default_diarization = settings.whisper_default_diarization
        self.default_model = settings.whisper_default_model
        self.default_format = settings.whisper_default_format

        # Upload storage
        self.uploaded_files: dict[str, dict] = {}

        # Redis
        self.redis_client = None
        if self.enable_caching:
            try:
                self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning("Redis connection failed, caching disabled", error=str(e))
                self.enable_caching = False

        logger.info(
            "AudioOrchestrator initialized",
            embedded_stt=True,
            integrated_diarization=True,
            language_detection=self.language_detection,
            speaker_diarization=self.speaker_diarization,
            caching=self.enable_caching,
        )

    # ---- Cache helpers ----

    async def get_cache_key(
        self, audio_data: bytes, operation: str, params: dict | None = None
    ) -> str:
        audio_hash = hashlib.sha256(audio_data).hexdigest()[:16]
        params_str = json.dumps(params or {}, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        return f"{operation}:{audio_hash}:{params_hash}"

    async def get_from_cache(self, key: str) -> dict | None:
        if not self.enable_caching or not self.redis_client:
            return None
        try:
            cached = await self.redis_client.get(key)
            if cached:
                logger.debug("Cache hit", key=key)
                return json.loads(cached)
        except Exception as e:
            logger.warning("Cache read error", key=key, error=str(e))
        return None

    async def set_cache(self, key: str, value: dict) -> None:
        if not self.enable_caching or not self.redis_client:
            return
        try:
            await self.redis_client.setex(key, self.cache_ttl, json.dumps(value))
            logger.debug("Cache stored", key=key)
        except Exception as e:
            logger.warning("Cache write error", key=key, error=str(e))

    # ---- Language detection ----

    async def detect_language_from_text(self, text: str) -> str:
        try:
            if not text.strip():
                return "unknown"
            detected = detect(text)
            lang_map = {"pl": "pl", "en": "en"}
            return lang_map.get(detected, "en")
        except Exception as e:
            logger.warning("Language detection failed", error=str(e))
            return "en"

    # ---- Whisper STT (embedded) ----

    async def call_whisper_api(
        self, audio_data: bytes, filename: str, language: str = "auto"
    ) -> dict[str, Any]:
        """Transcribe audio using embedded faster-whisper model."""
        try:
            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            try:
                lang = language if language != "auto" else None
                result = self.whisper_service.transcribe(tmp_path, lang)

                if language == "auto" and result.get("text"):
                    detected_lang = await self.detect_language_from_text(result["text"])
                    result["detected_language"] = detected_lang

                return result
            finally:
                with contextlib.suppress(Exception):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error("Embedded Whisper STT failed", error=str(e), filename=filename)
            raise HTTPException(status_code=500, detail=f"STT service error: {str(e)}") from e

    # ---- Diarization ----

    async def call_diarization_api(
        self, audio_data: bytes, _filename: str, _max_speakers: int = 10
    ) -> dict[str, Any]:
        """Call integrated PyAnnote speaker diarization."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            audio_buffer = io.BytesIO(audio_data)
            audio, sample_rate = librosa.load(audio_buffer, sr=None)

            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            sf.write(tmp_path, audio, sample_rate)

            logger.info(
                "Audio preprocessed for diarization",
                duration=len(audio) / sample_rate,
                sample_rate=sample_rate,
            )

            return self.diarization_service.perform_diarization(tmp_path)

        except Exception as e:
            logger.error("Diarization API call failed", error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Speaker diarization error: {str(e)}"
            ) from e

    # ---- Enhanced transcription ----

    async def process_enhanced_transcription(
        self,
        audio_data: bytes,
        filename: str,
        language: str = "auto",
        enable_speaker_diarization: bool | None = None,
        max_speakers: int = 10,
    ) -> dict[str, Any]:
        if enable_speaker_diarization is None:
            enable_speaker_diarization = self.speaker_diarization

        cache_key = await self.get_cache_key(
            audio_data,
            "enhanced_transcription",
            {
                "language": language,
                "speaker_diarization": enable_speaker_diarization,
                "max_speakers": max_speakers,
            },
        )

        cached_result = await self.get_from_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            if enable_speaker_diarization:
                diarization_result = await self.call_diarization_api(
                    audio_data, filename, max_speakers
                )
                stt_result = await self.call_whisper_api(audio_data, filename, language)
                detected_lang = await self.detect_language_from_text(stt_result.get("text", ""))
                result = await self._process_speaker_segments(
                    audio_data,
                    diarization_result,
                    stt_result,
                    detected_lang,
                    filename,
                )
            else:
                stt_result = await self.call_whisper_api(audio_data, filename, language)
                detected_lang = await self.detect_language_from_text(stt_result.get("text", ""))
                result = {
                    "text": stt_result.get("text", ""),
                    "language": detected_lang,
                    "segments": stt_result.get("segments", []),
                    "speaker_count": 1,
                    "speakers": {
                        "SPEAKER_00": {
                            "language": detected_lang,
                            "total_time": stt_result.get("duration", 0),
                            "segments_count": 1,
                        }
                    },
                    "phase": "basic_stt",
                }

            await self.set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error("Enhanced transcription failed", error=str(e))
            raise

    async def _process_speaker_segments(
        self,
        _audio_data: bytes,
        diarization_result: dict,
        stt_result: dict,
        detected_lang: str,
        _filename: str,
    ) -> dict[str, Any]:
        try:
            segments = diarization_result.get("segments", [])
            speakers_info = diarization_result.get("speakers", {})
            speaker_count = diarization_result.get("speaker_count", 1)

            detailed_segments = []

            if speaker_count == 1:
                seg_data = (
                    segments[0] if segments else {"start": 0.0, "end": 0, "speaker": "SPEAKER_00"}
                )
                detailed_segments.append(
                    {
                        "speaker": seg_data.get("speaker", "SPEAKER_00"),
                        "text": stt_result.get("text", ""),
                        "language": detected_lang,
                        "start": seg_data.get("start", 0.0),
                        "end": seg_data.get("end", 0),
                        "duration": seg_data.get("duration", 0),
                        "confidence": 0.95,
                    }
                )
            else:
                full_text = stt_result.get("text", "")
                text_map = self._distribute_text_to_segments(full_text, segments)

                for i, segment in enumerate(segments):
                    seg_text = text_map.get(i, "")
                    seg_lang = (
                        await self.detect_language_from_text(seg_text)
                        if seg_text
                        else detected_lang
                    )
                    detailed_segments.append(
                        {
                            "speaker": segment.get("speaker", f"SPEAKER_{i:02d}"),
                            "text": seg_text,
                            "language": seg_lang,
                            "start": float(segment.get("start", 0)),
                            "end": float(segment.get("end", 0)),
                            "duration": float(segment.get("duration", 0)),
                            "confidence": 0.92,
                        }
                    )

            enhanced_speakers = {}
            for speaker_id, info in speakers_info.items():
                sp_segs = [s for s in detailed_segments if s["speaker"] == speaker_id]
                sp_langs = list({s["language"] for s in sp_segs})
                enhanced_speakers[speaker_id] = {
                    "language": sp_langs[0] if sp_langs else detected_lang,
                    "languages": sp_langs,
                    "total_time": round(info.get("total_time", 0), 2),
                    "segments_count": info.get("segments_count", len(sp_segs)),
                    "avg_segment_duration": info.get("avg_segment_duration", 0),
                }

            return {
                "text": stt_result.get("text", ""),
                "language": detected_lang,
                "speaker_count": speaker_count,
                "speakers": enhanced_speakers,
                "segments": detailed_segments,
                "total_duration": round(diarization_result.get("total_duration", 0), 2),
                "phase": ("full_diarization" if speaker_count > 1 else "single_speaker"),
                "model": diarization_result.get("model", ""),
                "device": diarization_result.get("device", "cpu"),
            }

        except Exception as e:
            logger.error("Failed to process speaker segments", error=str(e))
            return {
                "text": stt_result.get("text", ""),
                "language": detected_lang,
                "speaker_count": 1,
                "speakers": {
                    "SPEAKER_00": {
                        "language": detected_lang,
                        "total_time": 0,
                        "segments_count": 1,
                    }
                },
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": stt_result.get("text", ""),
                        "language": detected_lang,
                        "start": 0.0,
                        "end": 0,
                        "duration": 0,
                        "confidence": 0.90,
                    }
                ],
                "phase": "fallback_single_speaker",
                "error": str(e),
            }

    def _distribute_text_to_segments(self, full_text: str, segments: list[dict]) -> dict[int, str]:
        if not full_text or not segments:
            return {}
        words = full_text.split()
        n = len(segments)
        per = max(1, len(words) // n)
        result = {}
        for i in range(n):
            start = i * per
            end = start + per if i < n - 1 else len(words)
            result[i] = " ".join(words[start:end])
        return result

    # ---- File upload management ----

    async def save_uploaded_file(self, file: UploadFile) -> dict[str, Any]:
        file_id = str(uuid.uuid4())
        filename = file.filename or f"audio_{file_id}"
        temp_dir = Path("/tmp/audio")
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_path = temp_dir / f"{file_id}_{filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        file_info = {
            "file_id": file_id,
            "filename": filename,
            "file_path": str(file_path),
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "processed": False,
        }
        self.uploaded_files[file_id] = file_info
        logger.info("File uploaded", file_id=file_id, filename=filename, size=len(content))
        return file_info

    async def get_uploaded_files_list(self) -> list[dict[str, Any]]:
        return list(self.uploaded_files.values())

    async def get_file_by_id(self, file_id: str) -> dict[str, Any] | None:
        return self.uploaded_files.get(file_id)

    async def remove_uploaded_file(self, file_id: str) -> bool:
        file_info = self.uploaded_files.get(file_id)
        if not file_info:
            return False
        try:
            if os.path.exists(file_info["file_path"]):
                os.remove(file_info["file_path"])
        except Exception:
            pass
        del self.uploaded_files[file_id]
        return True
