"""FastAPI route handlers for Voice Pipeline feature (translation, TTS, video)."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.whisper_unified.models.pipeline import JobResult, JobStatus, SynthesizeRequest

logger = structlog.get_logger()


def create_pipeline_router() -> APIRouter:
    """Create the API router for voice pipeline endpoints."""
    router = APIRouter()

    def _get_pipeline(request: Request):
        return request.app.state.pipeline_service

    def _save_upload(file: UploadFile) -> str:
        suffix = Path(file.filename or "upload").suffix or ".bin"
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")  # noqa: S108
        os.close(fd)
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return tmp

    @router.post("/v1/audio/translate", response_model=JobResult)
    async def translate_audio(
        request: Request,
        file: UploadFile = File(...),
        source_lang: str = Form(default="auto"),
        target_lang: str = Form(default="pl"),
        tts_voice: str = Form(default="alloy"),
    ) -> dict[str, Any]:
        """Transcribe audio -> translate text -> synthesize speech in target language."""
        pipeline = _get_pipeline(request)
        tmp = _save_upload(file)
        try:
            return await pipeline.translate_audio(tmp, source_lang, target_lang, tts_voice)
        except Exception as e:
            logger.error("Audio translation failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            Path(tmp).unlink(missing_ok=True)

    @router.post("/v1/video/translate", response_model=JobResult)
    async def translate_video(
        request: Request,
        file: UploadFile = File(...),
        source_lang: str = Form(default="auto"),
        target_lang: str = Form(default="pl"),
        tts_voice: str = Form(default="alloy"),
        keep_original_audio: bool = Form(default=False),
    ) -> dict[str, Any]:
        """Extract audio -> transcribe -> translate -> TTS -> merge back into video."""
        pipeline = _get_pipeline(request)
        tmp = _save_upload(file)
        try:
            return await pipeline.translate_video(
                tmp,
                source_lang,
                target_lang,
                tts_voice,
                keep_original_audio,
            )
        except Exception as e:
            logger.error("Video translation failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            Path(tmp).unlink(missing_ok=True)

    @router.post("/v1/subtitles/generate", response_model=JobResult)
    async def generate_subtitles(
        request: Request,
        file: UploadFile = File(...),
        source_lang: str = Form(default="auto"),
        target_lang: str = Form(default=None),
    ) -> dict[str, Any]:
        """Generate SRT subtitles from audio/video, optionally translated."""
        pipeline = _get_pipeline(request)
        tmp = _save_upload(file)
        try:
            return await pipeline.generate_subtitles(tmp, source_lang, target_lang)
        except Exception as e:
            logger.error("Subtitle generation failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            Path(tmp).unlink(missing_ok=True)

    @router.post("/v1/voice/learn", response_model=JobResult)
    async def learn_voice(
        request: Request,
        file: UploadFile = File(...),
        start_time: str = Form(default=None),
        end_time: str = Form(default=None),
    ) -> dict[str, Any]:
        """Extract voice sample from media file for future cloning."""
        pipeline = _get_pipeline(request)
        tmp = _save_upload(file)
        try:
            return await pipeline.learn_voice(tmp, start_time, end_time)
        except Exception as e:
            logger.error("Voice learning failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            Path(tmp).unlink(missing_ok=True)

    @router.post("/v1/voice/synthesize", response_model=JobResult)
    async def synthesize_voice(request: Request, req: SynthesizeRequest) -> JobResult:
        """Synthesize speech from text using configured TTS engine."""
        pipeline = _get_pipeline(request)
        try:
            output_name = f"synthesized_{req.language}.wav"
            output_path = str(pipeline.workspace / output_name)
            await pipeline.synthesize_speech(req.text, voice=req.voice, output_path=output_path)
            return JobResult(
                job_id="synth",
                status=JobStatus.completed,
                message="Speech synthesized",
                output_file=output_path,
            )
        except Exception as e:
            logger.error("Speech synthesis failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
