"""Voice pipeline service: translation, TTS, video processing, subtitle generation."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import structlog

from src.whisper_unified.models.pipeline import JobResult, JobStatus

if TYPE_CHECKING:
    from src.whisper_unified.config import Settings
    from src.whisper_unified.services.orchestrator import AudioOrchestrator

logger = structlog.get_logger()


class VoicePipelineService:
    """Orchestrates STT -> translation -> TTS pipeline for audio/video files."""

    def __init__(self, settings: Settings, orchestrator: AudioOrchestrator) -> None:
        self.settings = settings
        self.orchestrator = orchestrator
        self.ollama_url = settings.ollama_url
        self.tts_url = settings.tts_url
        self.tts_model = settings.tts_model
        self.tts_voice = settings.tts_voice
        self.ollama_model = settings.ollama_model
        self.workspace = Path(settings.workspace)
        self.timeout = httpx.Timeout(timeout=settings.job_timeout)

        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "samples").mkdir(parents=True, exist_ok=True)

        logger.info(
            "VoicePipelineService initialized",
            ollama_url=self.ollama_url,
            tts_url=self.tts_url,
            workspace=str(self.workspace),
        )

    # ---- External service health checks ----

    async def check_external_services(self) -> dict[str, bool]:
        """Check availability of Ollama and TTS services."""
        ollama, tts = await asyncio.gather(
            self._check_service(f"{self.ollama_url}/api/tags"),
            self._check_service(f"{self.tts_url}/v1/models"),
        )
        return {"ollama": ollama, "tts": tts}

    @staticmethod
    async def _check_service(url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    # ---- Internal STT (direct orchestrator call, no HTTP) ----

    async def _transcribe(self, audio_path: str, language: str = "auto") -> dict:
        """Transcribe audio using the embedded whisper service (no HTTP hop)."""
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        filename = Path(audio_path).name
        result = await self.orchestrator.call_whisper_api(audio_data, filename, language)
        return {
            "text": result.get("text", ""),
            "language": result.get("language", language),
            "segments": result.get("segments", []),
        }

    # ---- External service calls (Ollama, TTS) ----

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Ollama LLM."""
        if not text.strip():
            return ""

        prompt = (
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"Return ONLY the translation, no explanations.\n\n{text}"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    async def synthesize_speech(
        self, text: str, voice: str = "alloy", output_path: str = ""
    ) -> str:
        """Generate speech from text using OpenedAI Speech TTS."""
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav", dir="/tmp")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.tts_url}/v1/audio/speech",
                json={"model": self.tts_model, "voice": voice, "input": text},
            )
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)

        return output_path

    # ---- FFmpeg utilities ----

    @staticmethod
    def extract_audio_ffmpeg(video_path: str, output_path: str) -> None:
        """Extract audio track from video file (16kHz mono PCM)."""
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    @staticmethod
    def merge_audio_video_ffmpeg(
        video_path: str, audio_path: str, output_path: str, keep_original: bool = False
    ) -> None:
        """Replace or mix audio track in video."""
        if keep_original:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-i",
                    audio_path,
                    "-filter_complex",
                    "[0:a]volume=0.15[orig];[1:a]volume=1.0[new];"
                    "[orig][new]amix=inputs=2:duration=longest[out]",
                    "-map",
                    "0:v",
                    "-map",
                    "[out]",
                    "-c:v",
                    "copy",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-i",
                    audio_path,
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-shortest",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )

    # ---- Pipeline operations ----

    async def translate_audio(
        self,
        audio_path: str,
        source_lang: str = "auto",
        target_lang: str = "pl",
        tts_voice: str = "alloy",
    ) -> JobResult:
        """Full pipeline: STT -> translate -> TTS for audio file."""
        job_id = str(uuid.uuid4())[:8]
        logger.info("Job %s: translating audio %s → %s", job_id, source_lang, target_lang)

        try:
            result = await self._transcribe(audio_path, source_lang)
            transcript = result["text"]
            detected_lang = result["language"]

            if not transcript.strip():
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.failed,
                    message="Transcription returned empty text",
                    transcript="",
                )

            translated = await self.translate_text(transcript, detected_lang, target_lang)

            output_name = Path(audio_path).stem + f"_{target_lang}.wav"
            output_path = str(self.workspace / output_name)
            await self.synthesize_speech(translated, voice=tts_voice, output_path=output_path)

            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                message="Audio translation completed",
                output_file=output_path,
                transcript=transcript,
                translated_text=translated,
            )

        except Exception as e:
            logger.exception("Job %s failed", job_id, error=str(e))
            return JobResult(job_id=job_id, status=JobStatus.failed, message=str(e))

    async def translate_video(
        self,
        video_path: str,
        source_lang: str = "auto",
        target_lang: str = "pl",
        tts_voice: str = "alloy",
        keep_original_audio: bool = False,
    ) -> JobResult:
        """Full pipeline: extract audio -> STT -> translate -> TTS -> merge."""
        job_id = str(uuid.uuid4())[:8]
        logger.info("Job %s: translating video %s → %s", job_id, source_lang, target_lang)

        try:
            audio_tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
            self.extract_audio_ffmpeg(video_path, audio_tmp)

            result = await self._transcribe(audio_tmp, source_lang)
            transcript = result["text"]
            detected_lang = result["language"]

            if not transcript.strip():
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.failed,
                    message="Transcription returned empty text",
                )

            translated = await self.translate_text(transcript, detected_lang, target_lang)

            tts_tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
            await self.synthesize_speech(translated, voice=tts_voice, output_path=tts_tmp)

            output_name = Path(video_path).stem + f"_{target_lang}" + Path(video_path).suffix
            output_path = str(self.workspace / output_name)
            self.merge_audio_video_ffmpeg(
                video_path, tts_tmp, output_path, keep_original=keep_original_audio
            )

            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                message="Video translation completed",
                output_file=output_path,
                transcript=transcript,
                translated_text=translated,
            )

        except Exception as e:
            logger.exception("Job %s failed", job_id, error=str(e))
            return JobResult(job_id=job_id, status=JobStatus.failed, message=str(e))

    async def generate_subtitles(
        self,
        file_path: str,
        source_lang: str = "auto",
        target_lang: str | None = None,
    ) -> JobResult:
        """Generate SRT subtitles from audio/video file."""
        job_id = str(uuid.uuid4())[:8]

        try:
            ext = Path(file_path).suffix.lower()
            if ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                audio_tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
                self.extract_audio_ffmpeg(file_path, audio_tmp)
                audio_path = audio_tmp
            else:
                audio_path = file_path

            result = await self._transcribe(audio_path, source_lang)
            segments = result.get("segments", [])

            srt_lines = []
            for i, seg in enumerate(segments, 1):
                start = self._format_srt_time(seg.get("start", 0))
                end = self._format_srt_time(seg.get("end", 0))
                text = seg.get("text", "").strip()

                if target_lang and text:
                    text = await self.translate_text(text, source_lang, target_lang)

                srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

            srt_content = "\n".join(srt_lines)

            suffix = f"_{target_lang}" if target_lang else ""
            output_name = Path(file_path).stem + f"{suffix}.srt"
            output_path = str(self.workspace / output_name)
            with open(output_path, "w") as f:
                f.write(srt_content)

            transcript = " ".join(seg.get("text", "") for seg in segments)

            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                message="Subtitles generated",
                output_file=output_path,
                transcript=transcript,
            )

        except Exception as e:
            logger.exception("Job %s failed", job_id, error=str(e))
            return JobResult(job_id=job_id, status=JobStatus.failed, message=str(e))

    async def learn_voice(
        self,
        file_path: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> JobResult:
        """Extract voice sample from media file for cloning."""
        job_id = str(uuid.uuid4())[:8]

        try:
            output_name = Path(file_path).stem + "_voice_sample.wav"
            output_path = str(self.workspace / "samples" / output_name)

            cmd = ["ffmpeg", "-y", "-i", file_path]
            if start_time:
                cmd.extend(["-ss", start_time])
            if end_time:
                cmd.extend(["-to", end_time])
            cmd.extend(["-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", output_path])

            subprocess.run(cmd, check=True, capture_output=True)

            return JobResult(
                job_id=job_id,
                status=JobStatus.completed,
                message=f"Voice sample extracted to {output_path}",
                output_file=output_path,
            )

        except Exception as e:
            logger.exception("Job %s failed", job_id, error=str(e))
            return JobResult(job_id=job_id, status=JobStatus.failed, message=str(e))

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
