"""Tests for API routes."""

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_shows_services(self, client):
        response = client.get("/health")
        data = response.json()
        assert "services" in data
        assert "whisper_embedded" in data["services"]
        assert "diarization" in data["services"]

    def test_health_shows_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert "whisper_model" in data
        assert "whisper_device" in data


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Whisper Unified"
        assert data["version"] == "3.0.0"
        assert "endpoints" in data

    def test_root_features(self, client):
        response = client.get("/")
        data = response.json()
        assert "features" in data
        assert "embedded_stt" in data["features"]


class TestV1ApiInfo:
    def test_v1_returns_models(self, client):
        response = client.get("/v1")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2


class TestTranscriptionEndpoint:
    def test_transcribe_returns_text(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"language": "en", "force_auto": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data

    def test_transcribe_upload_only_mode(self, client_upload_only, sample_wav_bytes):
        """In upload-only mode, transcription endpoint uploads instead."""
        response = client_upload_only.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert "file_id" in data


class TestUploadsEndpoint:
    def test_list_uploads_empty(self, client):
        response = client.get("/v1/audio/uploads")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["uploaded_files"] == []

    def test_upload_file(self, client, sample_wav_bytes):
        response = client.post(
            "/v1/audio/uploads",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert data["filename"] == "test.wav"

    def test_upload_and_list(self, client, sample_wav_bytes):
        # Upload
        client.post(
            "/v1/audio/uploads",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        # List
        response = client.get("/v1/audio/uploads")
        data = response.json()
        assert data["count"] == 1

    def test_delete_upload(self, client, sample_wav_bytes):
        # Upload
        upload_resp = client.post(
            "/v1/audio/uploads",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        file_id = upload_resp.json()["file_id"]

        # Delete
        response = client.delete(f"/v1/audio/uploads/{file_id}")
        assert response.status_code == 200

        # Verify deleted
        response = client.get("/v1/audio/uploads")
        assert response.json()["count"] == 0

    def test_delete_nonexistent_returns_404(self, client):
        response = client.delete("/v1/audio/uploads/nonexistent-id")
        assert response.status_code == 404

    def test_get_upload_info(self, client, sample_wav_bytes):
        # Upload
        upload_resp = client.post(
            "/v1/audio/uploads",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )
        file_id = upload_resp.json()["file_id"]

        # Get info
        response = client.get(f"/v1/audio/uploads/{file_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["file_info"]["filename"] == "test.wav"

    def test_get_nonexistent_upload_returns_404(self, client):
        response = client.get("/v1/audio/uploads/nonexistent-id")
        assert response.status_code == 404
