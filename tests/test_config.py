"""Tests for Settings configuration."""

from unittest.mock import patch


class TestSettings:
    def test_default_values(self):
        """Settings loads with sane defaults."""
        with patch.dict(
            "os.environ",
            {"HOST": "0.0.0.0", "PORT": "8000"},
            clear=False,
        ):
            from src.whisper_unified.config import Settings

            settings = Settings(_env_file=None)
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.whisper_model_name == "Systran/faster-whisper-medium"
            assert settings.whisper_compute_type == "float16"
            assert settings.enable_caching is True
            assert settings.cache_ttl == 3600

    def test_override_from_env(self):
        """Settings can be overridden via environment variables."""
        with patch.dict(
            "os.environ",
            {
                "HOST": "127.0.0.1",
                "PORT": "9999",
                "WHISPER_MODEL_NAME": "Systran/faster-whisper-small",
                "WHISPER_COMPUTE_TYPE": "int8",
                "ENABLE_CACHING": "false",
                "CACHE_TTL": "600",
            },
        ):
            from src.whisper_unified.config import Settings

            settings = Settings(_env_file=None)
            assert settings.host == "127.0.0.1"
            assert settings.port == 9999
            assert settings.whisper_model_name == "Systran/faster-whisper-small"
            assert settings.whisper_compute_type == "int8"
            assert settings.enable_caching is False
            assert settings.cache_ttl == 600

    def test_bool_parsing(self):
        """Boolean env vars are parsed correctly."""
        with patch.dict(
            "os.environ",
            {
                "WHISPER_AUTO_TRANSCRIPTION": "false",
                "WHISPER_UPLOAD_ONLY_MODE": "true",
                "LANGUAGE_DETECTION": "false",
            },
        ):
            from src.whisper_unified.config import Settings

            settings = Settings(_env_file=None)
            assert settings.whisper_auto_transcription is False
            assert settings.whisper_upload_only_mode is True
            assert settings.language_detection is False

    def test_diarization_defaults(self):
        """Diarization-related settings have correct defaults."""
        with patch.dict("os.environ", {}, clear=False):
            from src.whisper_unified.config import Settings

            settings = Settings(_env_file=None)
            assert settings.pyannote_model == "pyannote/speaker-diarization-3.1"
            assert settings.huggingface_token == ""
            assert settings.whisper_default_diarization is True
