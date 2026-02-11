"""Tests for AudioOrchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAudioOrchestratorCache:
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, mock_settings):
        """Cache keys are deterministic for same input."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        key1 = await orch.get_cache_key(b"audio_data", "stt", {"lang": "en"})
                        key2 = await orch.get_cache_key(b"audio_data", "stt", {"lang": "en"})
                        assert key1 == key2

                        key3 = await orch.get_cache_key(b"different_data", "stt", {"lang": "en"})
                        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_disabled_returns_none(self, mock_settings):
        """When caching disabled, get_from_cache returns None."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        assert orch.enable_caching is False
                        result = await orch.get_from_cache("any_key")
                        assert result is None


class TestAudioOrchestratorLanguageDetection:
    @pytest.mark.asyncio
    async def test_detect_empty_text(self, mock_settings):
        """Empty text returns 'unknown'."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        lang = await orch.detect_language_from_text("")
                        assert lang == "unknown"

    @pytest.mark.asyncio
    async def test_detect_english_text(self, mock_settings):
        """English text is detected as 'en'."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        lang = await orch.detect_language_from_text(
                            "This is a test sentence in English language."
                        )
                        assert lang == "en"


class TestAudioOrchestratorFileUpload:
    @pytest.mark.asyncio
    async def test_save_and_list_files(self, mock_settings):
        """Files can be uploaded and listed."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)

                        mock_file = AsyncMock()
                        mock_file.filename = "test.wav"
                        mock_file.read.return_value = b"\x00" * 100

                        file_info = await orch.save_uploaded_file(mock_file)
                        assert file_info["filename"] == "test.wav"
                        assert file_info["size"] == 100
                        assert file_info["status"] == "uploaded"

                        files = await orch.get_uploaded_files_list()
                        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_remove_file(self, mock_settings):
        """Files can be removed."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)

                        mock_file = AsyncMock()
                        mock_file.filename = "test.wav"
                        mock_file.read.return_value = b"\x00" * 100

                        file_info = await orch.save_uploaded_file(mock_file)
                        file_id = file_info["file_id"]

                        success = await orch.remove_uploaded_file(file_id)
                        assert success is True

                        files = await orch.get_uploaded_files_list()
                        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_file(self, mock_settings):
        """Removing non-existent file returns False."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        result = await orch.remove_uploaded_file("nonexistent-id")
                        assert result is False


class TestTextDistribution:
    def test_distribute_text_evenly(self, mock_settings):
        """Text is distributed across segments proportionally."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        result = orch._distribute_text_to_segments(
                            "word1 word2 word3 word4",
                            [{"start": 0}, {"start": 1}],
                        )
                        assert len(result) == 2
                        assert "word1" in result[0]
                        assert "word4" in result[1]

    def test_distribute_empty_text(self, mock_settings):
        """Empty text returns empty dict."""
        with patch("src.whisper_unified.services.whisper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with patch("src.whisper_unified.services.whisper.WhisperModel"):
                with patch("src.whisper_unified.services.diarization.torch") as dt:
                    dt.cuda.is_available.return_value = False
                    with patch("src.whisper_unified.services.diarization.Pipeline"):
                        from src.whisper_unified.services.orchestrator import (
                            AudioOrchestrator,
                        )

                        orch = AudioOrchestrator(mock_settings)
                        result = orch._distribute_text_to_segments("", [{"start": 0}])
                        assert result == {}
