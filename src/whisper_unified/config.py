"""Centralized configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Service
    host: str = "0.0.0.0"
    port: int = 8000

    # Whisper STT
    whisper_model_name: str = "Systran/faster-whisper-medium"
    whisper_language: str = "auto"
    whisper_compute_type: str = "float16"

    # GPU
    cuda_visible_devices: str = "0"

    # Speaker Diarization
    enable_speaker_diarization: bool = True
    speaker_diarization: bool = False
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    huggingface_token: str = ""

    # Redis
    redis_url: str = "redis://whisper-redis:6379"
    enable_caching: bool = True
    cache_ttl: int = 3600

    # Language detection
    language_detection: bool = True
    whisper_supported_languages: str = "pl,en,auto"

    # Auto-transcription control
    whisper_auto_transcription: bool = True
    whisper_upload_only_mode: bool = False
    whisper_require_explicit_start: bool = False
    whisper_show_file_info_on_upload: bool = True

    # Default processing parameters
    whisper_default_language: str = "auto"
    whisper_default_diarization: bool = True
    whisper_default_model: str = "whisper-multilang"
    whisper_default_format: str = "detailed_json"
