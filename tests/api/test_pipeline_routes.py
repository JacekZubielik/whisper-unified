"""Tests for Voice Pipeline API routes."""

from unittest.mock import AsyncMock

from src.whisper_unified.models.pipeline import JobResult, JobStatus


class TestHealthWithPipeline:
    def test_health_includes_pipeline_services(self, client_with_pipeline):
        """Health endpoint shows ollama and tts status when pipeline enabled."""
        # Mock external service checks to avoid real HTTP calls
        pipeline = client_with_pipeline.app.state.pipeline_service
        pipeline.check_external_services = AsyncMock(return_value={"ollama": True, "tts": False})

        response = client_with_pipeline.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "ollama" in data["services"]
        assert "tts" in data["services"]
        assert data["services"]["ollama"] is True
        assert data["services"]["tts"] is False


class TestRootWithPipeline:
    def test_root_shows_pipeline_endpoints(self, client_with_pipeline):
        """Root endpoint lists pipeline endpoints when enabled."""
        response = client_with_pipeline.get("/")
        data = response.json()
        assert "audio_translate" in data["endpoints"]
        assert "video_translate" in data["endpoints"]
        assert "subtitles" in data["endpoints"]
        assert "voice_learn" in data["endpoints"]
        assert "voice_synthesize" in data["endpoints"]

    def test_root_shows_pipeline_feature(self, client_with_pipeline):
        """Root endpoint shows voice_pipeline in features."""
        response = client_with_pipeline.get("/")
        data = response.json()
        assert data["features"]["voice_pipeline"] is True


class TestTranslateAudioEndpoint:
    def test_returns_job_result(self, client_with_pipeline, sample_wav_bytes):
        """POST /v1/audio/translate returns JobResult."""
        pipeline = client_with_pipeline.app.state.pipeline_service
        pipeline.translate_audio = AsyncMock(
            return_value=JobResult(
                job_id="test123",
                status=JobStatus.completed,
                message="Audio translation completed",
                output_file="/workspace/test_pl.wav",
                transcript="Hello world",
                translated_text="Cześć świecie",
            )
        )

        response = client_with_pipeline.post(
            "/v1/audio/translate",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"source_lang": "en", "target_lang": "pl"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["transcript"] == "Hello world"
        assert data["translated_text"] == "Cześć świecie"


class TestSubtitlesEndpoint:
    def test_returns_job_result(self, client_with_pipeline, sample_wav_bytes):
        """POST /v1/subtitles/generate returns JobResult."""
        pipeline = client_with_pipeline.app.state.pipeline_service
        pipeline.generate_subtitles = AsyncMock(
            return_value=JobResult(
                job_id="sub123",
                status=JobStatus.completed,
                message="Subtitles generated",
                output_file="/workspace/test.srt",
                transcript="Hello world",
            )
        )

        response = client_with_pipeline.post(
            "/v1/subtitles/generate",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"source_lang": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["output_file"].endswith(".srt")


class TestSynthesizeEndpoint:
    def test_returns_job_result(self, client_with_pipeline):
        """POST /v1/voice/synthesize returns JobResult."""
        pipeline = client_with_pipeline.app.state.pipeline_service
        pipeline.synthesize_speech = AsyncMock(return_value="/workspace/synthesized_pl.wav")

        response = client_with_pipeline.post(
            "/v1/voice/synthesize",
            json={"text": "Cześć świecie", "language": "pl", "voice": "alloy"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["message"] == "Speech synthesized"


class TestVoiceLearnEndpoint:
    def test_returns_job_result(self, client_with_pipeline, sample_wav_bytes):
        """POST /v1/voice/learn returns JobResult."""
        pipeline = client_with_pipeline.app.state.pipeline_service
        pipeline.learn_voice = AsyncMock(
            return_value=JobResult(
                job_id="learn123",
                status=JobStatus.completed,
                message="Voice sample extracted",
                output_file="/workspace/samples/test_voice_sample.wav",
            )
        )

        response = client_with_pipeline.post(
            "/v1/voice/learn",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
