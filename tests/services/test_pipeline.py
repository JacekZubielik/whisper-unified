"""Tests for VoicePipelineService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.whisper_unified.models.pipeline import JobStatus


class TestVoicePipelineInit:
    def test_creates_workspace(self, app_with_pipeline):
        """Pipeline service creates workspace directory on init."""
        pipeline = app_with_pipeline.state.pipeline_service
        assert pipeline.workspace.exists()
        assert (pipeline.workspace / "samples").exists()

    def test_stores_settings(self, app_with_pipeline):
        """Pipeline service stores config from settings."""
        pipeline = app_with_pipeline.state.pipeline_service
        assert pipeline.ollama_url == "http://localhost:11434"
        assert pipeline.tts_url == "http://localhost:8000"
        assert pipeline.ollama_model == "test-model"


class TestTranscribeInternal:
    @pytest.mark.asyncio
    async def test_calls_orchestrator_directly(self, app_with_pipeline, tmp_path):
        """_transcribe() calls orchestrator.call_whisper_api() without HTTP."""
        pipeline = app_with_pipeline.state.pipeline_service

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        pipeline.orchestrator.call_whisper_api = AsyncMock(return_value={
            "text": "Hello world",
            "language": "en",
            "segments": [{"start": 0, "end": 5, "text": "Hello world"}],
        })

        result = await pipeline._transcribe(str(audio_file), "auto")

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert len(result["segments"]) == 1
        pipeline.orchestrator.call_whisper_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_text(self, app_with_pipeline, tmp_path):
        """_transcribe() returns empty text when orchestrator returns nothing."""
        pipeline = app_with_pipeline.state.pipeline_service

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        pipeline.orchestrator.call_whisper_api = AsyncMock(return_value={})

        result = await pipeline._transcribe(str(audio_file))
        assert result["text"] == ""


class TestTranslateText:
    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, app_with_pipeline):
        """translate_text() returns empty string for empty input."""
        pipeline = app_with_pipeline.state.pipeline_service
        result = await pipeline.translate_text("", "en", "pl")
        assert result == ""

    @pytest.mark.asyncio
    async def test_calls_ollama(self, app_with_pipeline):
        """translate_text() calls Ollama /api/generate."""
        pipeline = app_with_pipeline.state.pipeline_service

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Cześć świecie"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await pipeline.translate_text("Hello world", "en", "pl")
            assert result == "Cześć świecie"
            mock_client.post.assert_called_once()


class TestSynthesizeSpeech:
    @pytest.mark.asyncio
    async def test_saves_audio_file(self, app_with_pipeline, tmp_path):
        """synthesize_speech() writes TTS response to file."""
        pipeline = app_with_pipeline.state.pipeline_service
        output_path = str(tmp_path / "output.wav")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake audio data"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await pipeline.synthesize_speech("Cześć", output_path=output_path)
            assert result == output_path

            with open(output_path, "rb") as f:
                assert f.read() == b"fake audio data"


class TestTranslateAudio:
    @pytest.mark.asyncio
    async def test_empty_transcription_returns_failed(self, app_with_pipeline, tmp_path):
        """translate_audio() returns failed if transcription is empty."""
        pipeline = app_with_pipeline.state.pipeline_service

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        pipeline.orchestrator.call_whisper_api = AsyncMock(return_value={
            "text": "", "language": "en", "segments": [],
        })

        result = await pipeline.translate_audio(str(audio_file))
        assert result.status == JobStatus.failed
        assert "empty" in result.message.lower()


class TestExternalHealthChecks:
    @pytest.mark.asyncio
    async def test_returns_service_status(self, app_with_pipeline):
        """check_external_services() returns ollama and tts status."""
        pipeline = app_with_pipeline.state.pipeline_service

        with patch.object(pipeline, "_check_service", new_callable=AsyncMock) as mock:
            mock.side_effect = [True, False]
            result = await pipeline.check_external_services()
            assert result == {"ollama": True, "tts": False}


class TestFormatSrtTime:
    def test_zero(self):
        from src.whisper_unified.services.pipeline import VoicePipelineService

        assert VoicePipelineService._format_srt_time(0) == "00:00:00,000"

    def test_with_milliseconds(self):
        from src.whisper_unified.services.pipeline import VoicePipelineService

        assert VoicePipelineService._format_srt_time(3.5) == "00:00:03,500"

    def test_minutes_and_hours(self):
        from src.whisper_unified.services.pipeline import VoicePipelineService

        assert VoicePipelineService._format_srt_time(3661.123) == "01:01:01,123"
