"""FastAPI route handlers for Whisper Unified API."""

import os
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = structlog.get_logger()


def create_router() -> APIRouter:
    """Create the API router with all endpoints."""
    router = APIRouter()

    def _get_orchestrator(request: Request):
        return request.app.state.orchestrator

    @router.get("/health")
    async def health_check(request: Request) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        status = {
            "status": "healthy",
            "services": {
                "whisper_embedded": orchestrator.whisper_service.model is not None,
                "diarization": orchestrator.diarization_service.pipeline is not None,
                "redis": False,
            },
            "whisper_model": orchestrator.whisper_service.model_name,
            "whisper_device": orchestrator.whisper_service.device,
        }

        if orchestrator.enable_caching and orchestrator.redis_client:
            try:
                await orchestrator.redis_client.ping()
                status["services"]["redis"] = True
            except Exception:
                pass
        else:
            status["services"]["redis"] = True  # Not required

        if hasattr(request.app.state, "pipeline_service"):
            ext = await request.app.state.pipeline_service.check_external_services()
            status["services"]["ollama"] = ext["ollama"]
            status["services"]["tts"] = ext["tts"]

        return status

    @router.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(None),
        language: str = Form(None),
        response_format: str = Form("json"),
        force_auto: bool = Form(False),
    ) -> dict[str, Any]:
        """OpenAI-compatible transcription endpoint."""
        orchestrator = _get_orchestrator(request)
        try:
            if not orchestrator.auto_transcription or orchestrator.upload_only_mode:
                if not force_auto:
                    file_info = await orchestrator.save_uploaded_file(file)
                    return {
                        "status": "uploaded",
                        "message": "File uploaded. Use /v1/audio/start-transcription to process.",
                        "file_id": file_info["file_id"],
                        "filename": file_info["filename"],
                        "size": file_info["size"],
                    }

            audio_data = await file.read()
            lang = language if language else "auto"
            result = await orchestrator.call_whisper_api(audio_data, file.filename, lang)

            if response_format == "json":
                return {"text": result.get("text", "")}
            return result

        except Exception as e:
            logger.error("Transcription failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/v1/audio/transcriptions/enhanced")
    async def enhanced_transcription(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-multilang"),
        language: str = Form("auto"),
        enable_speaker_diarization: bool = Form(False),
        max_speakers: int = Form(10),
        min_speaker_duration: float = Form(1.0),
        response_format: str = Form("detailed_json"),
        force_auto: bool = Form(False),
    ) -> dict[str, Any]:
        """Enhanced transcription with speaker diarization."""
        orchestrator = _get_orchestrator(request)
        try:
            if not orchestrator.auto_transcription or orchestrator.upload_only_mode:
                if not force_auto:
                    file_info = await orchestrator.save_uploaded_file(file)
                    return {
                        "status": "uploaded",
                        "message": "File uploaded. Use /v1/audio/start-transcription to process.",
                        "file_id": file_info["file_id"],
                        "filename": file_info["filename"],
                        "size": file_info["size"],
                    }

            audio_data = await file.read()
            return await orchestrator.process_enhanced_transcription(
                audio_data=audio_data,
                filename=file.filename,
                language=language,
                enable_speaker_diarization=enable_speaker_diarization,
                max_speakers=max_speakers,
            )

        except Exception as e:
            logger.error("Enhanced transcription failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/v1/audio/language-detection")
    async def detect_language(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        """Language detection endpoint."""
        orchestrator = _get_orchestrator(request)
        try:
            audio_data = await file.read()
            result = await orchestrator.call_whisper_api(audio_data, file.filename, "auto")
            return {
                "language": result.get("detected_language", result.get("language", "en")),
                "language_probability": result.get("language_probability", 0),
                "text_sample": result.get("text", "")[:100],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/v1/audio/speaker-diarization")
    async def speaker_diarization(
        request: Request,
        file: UploadFile = File(...),
        max_speakers: int = Form(10),
        min_duration: float = Form(1.0),
    ) -> dict[str, Any]:
        """Speaker diarization only endpoint."""
        orchestrator = _get_orchestrator(request)
        try:
            audio_data = await file.read()
            return await orchestrator.call_diarization_api(audio_data, file.filename, max_speakers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/v1/audio/uploads")
    async def list_uploads(request: Request) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        files = await orchestrator.get_uploaded_files_list()
        return {"status": "success", "uploaded_files": files, "count": len(files)}

    @router.post("/v1/audio/uploads")
    async def upload_file(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload audio file without processing."""
        orchestrator = _get_orchestrator(request)
        file_info = await orchestrator.save_uploaded_file(file)
        return {
            "status": "uploaded",
            "file_id": file_info["file_id"],
            "filename": file_info["filename"],
            "size": file_info["size"],
        }

    @router.post("/v1/audio/start-transcription")
    async def start_manual_transcription(
        request: Request,
        file_id: str = Form(...),
        model: str = Form("whisper-multilang"),
        language: str = Form("auto"),
        enable_speaker_diarization: bool = Form(True),
        max_speakers: int = Form(10),
        response_format: str = Form("detailed_json"),
    ) -> dict[str, Any]:
        """Start transcription for previously uploaded file."""
        orchestrator = _get_orchestrator(request)
        try:
            file_info = await orchestrator.get_file_by_id(file_id)
            if not file_info:
                raise HTTPException(status_code=404, detail=f"File {file_id} not found")
            if file_info["processed"]:
                raise HTTPException(status_code=400, detail="File already processed")
            if not os.path.exists(file_info["file_path"]):
                raise HTTPException(status_code=404, detail="Physical file not found")

            with open(file_info["file_path"], "rb") as f:
                audio_data = f.read()

            file_info["status"] = "processing"
            file_info["processing_start"] = datetime.now().isoformat()

            if response_format == "detailed_json" or enable_speaker_diarization:
                result = await orchestrator.process_enhanced_transcription(
                    audio_data=audio_data,
                    filename=file_info["filename"],
                    language=language,
                    enable_speaker_diarization=enable_speaker_diarization,
                    max_speakers=max_speakers,
                )
            else:
                result = await orchestrator.call_whisper_api(
                    audio_data, file_info["filename"], language
                )
                if response_format == "json":
                    result = {"text": result.get("text", "")}

            file_info["status"] = "completed"
            file_info["processed"] = True
            file_info["processing_end"] = datetime.now().isoformat()
            file_info["result"] = result

            return {
                "status": "completed",
                "file_id": file_id,
                "filename": file_info["filename"],
                "result": result,
            }

        except HTTPException:
            raise
        except Exception as e:
            file_info = await orchestrator.get_file_by_id(file_id)
            if file_info:
                file_info["status"] = "failed"
                file_info["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.delete("/v1/audio/uploads/{file_id}")
    async def delete_upload(request: Request, file_id: str) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        success = await orchestrator.remove_uploaded_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        return {"status": "success", "message": f"File {file_id} deleted"}

    @router.get("/v1/audio/uploads/{file_id}")
    async def get_upload_info(request: Request, file_id: str) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        file_info = await orchestrator.get_file_by_id(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        return {"status": "success", "file_info": file_info}

    @router.get("/v1")
    async def v1_api_info(request: Request) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": "whisper-1",
                    "object": "model",
                    "owned_by": "openai-internal",
                },
                {
                    "id": "whisper-multilang",
                    "object": "model",
                    "owned_by": "systran",
                },
            ],
            "service_info": {
                "service": "Whisper Unified",
                "version": request.app.version,
                "api_compatibility": "OpenAI v1",
                "embedded_stt": True,
                "whisper_model": orchestrator.whisper_service.model_name,
            },
        }

    @router.get("/")
    async def root(request: Request) -> dict[str, Any]:
        orchestrator = _get_orchestrator(request)
        endpoints = {
            "health": "/health",
            "transcription": "/v1/audio/transcriptions",
            "enhanced": "/v1/audio/transcriptions/enhanced",
            "language_detection": "/v1/audio/language-detection",
            "speaker_diarization": "/v1/audio/speaker-diarization",
            "uploads": "/v1/audio/uploads",
            "manual_transcription": "/v1/audio/start-transcription",
        }

        if hasattr(request.app.state, "pipeline_service"):
            endpoints.update(
                {
                    "audio_translate": "/v1/audio/translate",
                    "video_translate": "/v1/video/translate",
                    "subtitles": "/v1/subtitles/generate",
                    "voice_learn": "/v1/voice/learn",
                    "voice_synthesize": "/v1/voice/synthesize",
                }
            )

        return {
            "service": "Whisper Unified",
            "version": request.app.version,
            "description": "Embedded STT + Speaker Diarization + Voice Pipeline",
            "features": {
                "embedded_stt": True,
                "language_detection": orchestrator.language_detection,
                "speaker_diarization": orchestrator.speaker_diarization,
                "caching": orchestrator.enable_caching,
                "voice_pipeline": hasattr(request.app.state, "pipeline_service"),
            },
            "endpoints": endpoints,
        }

    return router
