"""Tests for IntegratedDiarizationService."""

from unittest.mock import MagicMock, patch

import pytest


class TestIntegratedDiarizationService:
    def test_init(self, mock_settings):
        """Service initializes with settings."""
        with patch("src.whisper_unified.services.diarization.torch"):
            with patch("src.whisper_unified.services.diarization.Pipeline"):
                from src.whisper_unified.services.diarization import (
                    IntegratedDiarizationService,
                )

                service = IntegratedDiarizationService(mock_settings)
                assert service.pipeline is None
                assert service.device is None
                assert service.hf_token == ""

    @pytest.mark.asyncio
    async def test_initialize_without_token_uses_mock(self, mock_settings):
        """Without HF token, falls back to mock pipeline."""
        with patch("src.whisper_unified.services.diarization.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.diarization.Pipeline") as mock_pipeline:
                mock_pipeline.from_pretrained.side_effect = Exception("No token")

                from src.whisper_unified.services.diarization import (
                    IntegratedDiarizationService,
                )

                service = IntegratedDiarizationService(mock_settings)
                success = await service.initialize()

                assert success is True
                assert service.pipeline is not None
                assert service.device == "cpu"

    def test_perform_diarization(self, mock_settings, mock_diarization_pipeline):
        """Diarization returns expected result structure."""
        with patch("src.whisper_unified.services.diarization.torch"):
            with patch("src.whisper_unified.services.diarization.Pipeline"):
                from src.whisper_unified.services.diarization import (
                    IntegratedDiarizationService,
                )

                service = IntegratedDiarizationService(mock_settings)
                service.pipeline = mock_diarization_pipeline
                service.device = "cpu"

                # Create a temp file for the test
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(b"\x00" * 100)
                    tmp_path = tmp.name

                result = service.perform_diarization(tmp_path)

                assert "speaker_count" in result
                assert "segments" in result
                assert "speakers" in result
                assert result["speaker_count"] == 2
                assert len(result["segments"]) == 2

    def test_mock_pipeline_creation(self, mock_settings):
        """Mock pipeline can be created and used."""
        with patch("src.whisper_unified.services.diarization.torch"):
            with patch("src.whisper_unified.services.diarization.Pipeline"):
                from src.whisper_unified.services.diarization import (
                    IntegratedDiarizationService,
                )

                service = IntegratedDiarizationService(mock_settings)
                mock_pipeline = service._create_mock_pipeline()

                assert mock_pipeline is not None
                assert hasattr(mock_pipeline, "to")
                # to() should return self
                assert mock_pipeline.to("cpu") is mock_pipeline
