"""Pydantic models for the voice pipeline feature."""

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a pipeline job."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class SynthesizeRequest(BaseModel):
    """Request body for POST /v1/voice/synthesize."""

    text: str = Field(..., description="Text to synthesize")
    language: str = Field(default="pl", description="Language code")
    voice: str = Field(default="alloy", description="Voice name")


class JobResult(BaseModel):
    """Result of a pipeline job."""

    job_id: str
    status: JobStatus
    message: str
    output_file: str | None = None
    transcript: str | None = None
    translated_text: str | None = None
    duration: float | None = None
