"""Shared test fixtures for whisper-unified."""

import struct
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-populate sys.modules with mock ML dependencies.
# These packages live in the optional 'ml' group and are NOT installed in the
# CI / test environment.  Without this block every `import torch` (etc.) at
# module level in src/ would raise ImportError before patches can be applied.
# ---------------------------------------------------------------------------
_MOCK_ML_MODULES = [
    "torch",
    "torch.cuda",
    "torch.nn",
    "torchaudio",
    "faster_whisper",
    "pyannote",
    "pyannote.audio",
    "librosa",
    "soundfile",
    "ctranslate2",
    "numba",
]
for _mod in _MOCK_ML_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


@pytest.fixture
def mock_settings():
    """Settings with test defaults (no GPU, no Redis, no real models)."""
    with patch.dict(
        "os.environ",
        {
            "HOST": "127.0.0.1",
            "PORT": "8000",
            "WHISPER_MODEL_NAME": "tiny",
            "WHISPER_COMPUTE_TYPE": "int8",
            "ENABLE_CACHING": "false",
            "SPEAKER_DIARIZATION": "false",
            "ENABLE_SPEAKER_DIARIZATION": "false",
            "HUGGINGFACE_TOKEN": "",
            "REDIS_URL": "redis://localhost:6380",
            "WHISPER_AUTO_TRANSCRIPTION": "true",
            "WHISPER_UPLOAD_ONLY_MODE": "false",
            "WHISPER_REQUIRE_EXPLICIT_START": "false",
        },
    ):
        from src.whisper_unified.config import Settings

        yield Settings(_env_file=None)


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper WhisperModel with fake transcribe results."""
    model = MagicMock()

    segment = MagicMock()
    segment.start = 0.0
    segment.end = 5.0
    segment.text = " Test transcription text."

    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.95
    info.duration = 5.0

    model.transcribe.return_value = ([segment], info)
    return model


@pytest.fixture
def mock_diarization_pipeline():
    """Mock PyAnnote Pipeline with fake diarization results."""
    pipeline = MagicMock()

    turn1 = MagicMock()
    turn1.start = 0.0
    turn1.end = 3.0
    turn2 = MagicMock()
    turn2.start = 3.0
    turn2.end = 5.0

    diarization = MagicMock()
    diarization.itertracks.return_value = [
        (turn1, None, "SPEAKER_00"),
        (turn2, None, "SPEAKER_01"),
    ]
    pipeline.return_value = diarization
    pipeline.to.return_value = pipeline
    return pipeline


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    redis = AsyncMock()
    redis.ping.return_value = True
    redis.get.return_value = None
    redis.setex.return_value = True
    return redis


@pytest.fixture
def sample_wav_bytes():
    """Minimal valid WAV file (44 bytes header + 2 bytes of silence)."""
    # WAV header for 16-bit mono PCM at 16000 Hz, 1 sample
    data_size = 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        1,  # mono
        16000,  # sample rate
        32000,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + b"\x00\x00"


def _create_test_app(
    mock_whisper_model: MagicMock,
    mock_diarization_pipeline: MagicMock,
    mock_redis: AsyncMock,
    env_overrides: dict[str, str] = None,
):
    """Helper to create a test FastAPI app with mocked services."""
    env = {
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "WHISPER_MODEL_NAME": "tiny",
        "WHISPER_COMPUTE_TYPE": "int8",
        "ENABLE_CACHING": "false",
        "SPEAKER_DIARIZATION": "false",
        "ENABLE_SPEAKER_DIARIZATION": "false",
        "HUGGINGFACE_TOKEN": "",
        "REDIS_URL": "redis://localhost:6380",
        "WHISPER_AUTO_TRANSCRIPTION": "true",
        "WHISPER_UPLOAD_ONLY_MODE": "false",
        "WHISPER_REQUIRE_EXPLICIT_START": "false",
    }
    if env_overrides:
        env.update(env_overrides)

    with patch.dict("os.environ", env):
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with (
                patch(
                    "src.whisper_unified.services.whisper.WhisperModel",
                    return_value=mock_whisper_model,
                ),
                patch("src.whisper_unified.services.diarization.torch") as mock_dtorch,
            ):
                mock_dtorch.cuda.is_available.return_value = False

                with patch(
                    "src.whisper_unified.services.diarization.Pipeline"
                ) as mock_pipeline_cls:
                    mock_pipeline_cls.from_pretrained.return_value = mock_diarization_pipeline

                    from src.whisper_unified.api.app import create_app
                    from src.whisper_unified.config import Settings

                    settings = Settings(_env_file=None)
                    app = create_app(settings)

                    # Pre-load mocked services — bypass lifespan which
                    # only runs inside a TestClient context manager.
                    orch = app.state.orchestrator
                    orch.whisper_service.model = mock_whisper_model
                    orch.diarization_service.pipeline = mock_diarization_pipeline
                    orch.diarization_service.device = "cpu"

                    return app


@pytest.fixture
def app(mock_whisper_model, mock_diarization_pipeline, mock_redis):
    """FastAPI test app with all ML services mocked."""
    return _create_test_app(mock_whisper_model, mock_diarization_pipeline, mock_redis)


@pytest.fixture
def app_upload_only(mock_whisper_model, mock_diarization_pipeline, mock_redis):
    """FastAPI test app in upload-only mode."""
    return _create_test_app(
        mock_whisper_model,
        mock_diarization_pipeline,
        mock_redis,
        env_overrides={
            "WHISPER_AUTO_TRANSCRIPTION": "false",
            "WHISPER_UPLOAD_ONLY_MODE": "true",
        },
    )


@pytest.fixture
def client(app):
    """TestClient for the mocked FastAPI app."""
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def client_upload_only(app_upload_only):
    """TestClient for upload-only mode."""
    from fastapi.testclient import TestClient

    return TestClient(app_upload_only)
