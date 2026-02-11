"""Tests for Voice Pipeline Pydantic models."""

import pytest

from src.whisper_unified.models.pipeline import JobResult, JobStatus, SynthesizeRequest


class TestJobStatus:
    def test_enum_values(self):
        assert JobStatus.pending == "pending"
        assert JobStatus.processing == "processing"
        assert JobStatus.completed == "completed"
        assert JobStatus.failed == "failed"

    def test_enum_is_str(self):
        assert isinstance(JobStatus.completed, str)


class TestJobResult:
    def test_minimal_creation(self):
        result = JobResult(job_id="abc123", status=JobStatus.completed, message="OK")
        assert result.job_id == "abc123"
        assert result.status == JobStatus.completed
        assert result.output_file is None
        assert result.transcript is None
        assert result.translated_text is None
        assert result.duration is None

    def test_full_creation(self):
        result = JobResult(
            job_id="abc123",
            status=JobStatus.completed,
            message="Done",
            output_file="/tmp/out.wav",
            transcript="hello",
            translated_text="cześć",
            duration=5.0,
        )
        assert result.output_file == "/tmp/out.wav"
        assert result.transcript == "hello"
        assert result.translated_text == "cześć"
        assert result.duration == 5.0

    def test_failed_status(self):
        result = JobResult(job_id="x", status=JobStatus.failed, message="error occurred")
        assert result.status == "failed"


class TestSynthesizeRequest:
    def test_text_is_required(self):
        with pytest.raises(Exception):
            SynthesizeRequest()

    def test_defaults(self):
        req = SynthesizeRequest(text="hello world")
        assert req.text == "hello world"
        assert req.language == "pl"
        assert req.voice == "alloy"

    def test_custom_values(self):
        req = SynthesizeRequest(text="hi", language="en", voice="nova")
        assert req.language == "en"
        assert req.voice == "nova"
