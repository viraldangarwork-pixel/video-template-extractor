import io
from fastapi.testclient import TestClient
from vtpl.web.app import app

client = TestClient(app)


def test_upload_rejection_for_unsupported_format():
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_upload_rejection_for_empty_file():
    response = client.post(
        "/api/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_rejection_for_large_content_length():
    # Simulate a 1000MB file via Content-Length header
    fake_video = io.BytesIO(b"\x00" * 1024)
    response = client.post(
        "/api/upload",
        files={"file": ("huge_clip.mp4", fake_video, "video/mp4")},
        headers={"Content-Length": str(1000 * 1024 * 1024)},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]


def test_upload_valid_small_clip():
    small_mp4 = io.BytesIO(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
    response = client.post(
        "/api/upload",
        files={"file": ("valid.mp4", small_mp4, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
