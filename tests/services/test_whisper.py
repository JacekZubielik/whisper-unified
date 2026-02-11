"""Tests for EmbeddedWhisperService."""

from unittest.mock import patch

import pytest


class TestEmbeddedWhisperService:
    def test_init_cpu_fallback(self, mock_settings):
        """Service falls back to CPU when CUDA unavailable."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                assert service.device == "cpu"
                assert service.model_name == "tiny"

    def test_init_cuda(self, mock_settings):
        """Service uses CUDA when available."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                assert service.device == "cuda"

    def test_load_model(self, mock_settings, mock_whisper_model):
        """Model can be loaded successfully."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch(
                "src.whisper_unified.services.whisper.WhisperModel",
                return_value=mock_whisper_model,
            ):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                assert service.model is None
                service.load_model()
                assert service.model is not None

    def test_transcribe_without_model_raises(self, mock_settings):
        """Transcribe raises RuntimeError if model not loaded."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                with pytest.raises(RuntimeError, match="Whisper model not loaded"):
                    service.transcribe("/fake/path.wav")

    def test_transcribe_returns_result(self, mock_settings, mock_whisper_model):
        """Transcribe returns expected structure."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch(
                "src.whisper_unified.services.whisper.WhisperModel",
                return_value=mock_whisper_model,
            ):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                service.load_model()
                result = service.transcribe("/fake/path.wav")

                assert "text" in result
                assert "segments" in result
                assert "language" in result
                assert result["text"] == "Test transcription text."
                assert result["language"] == "en"
                assert result["language_probability"] == 0.95
                assert result["duration"] == 5.0

    def test_transcribe_auto_language(self, mock_settings, mock_whisper_model):
        """Transcribe with 'auto' language passes None to model."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch(
                "src.whisper_unified.services.whisper.WhisperModel",
                return_value=mock_whisper_model,
            ):
                from src.whisper_unified.services.whisper import EmbeddedWhisperService

                service = EmbeddedWhisperService(mock_settings)
                service.load_model()
                service.transcribe("/fake/path.wav", language="auto")

                mock_whisper_model.transcribe.assert_called_once()
                call_args = mock_whisper_model.transcribe.call_args
                assert call_args[1]["language"] is None
