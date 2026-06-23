import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

class MockCLIP(torch.nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.dim = dim
    def encode_text(self, tokens):
        return torch.randn(1, self.dim)
    def encode_image(self, img):
        return torch.randn(1, self.dim)

@pytest.fixture(scope="function")
def test_app(tmp_path, monkeypatch):
    # Define temporary directories for absolute isolation
    test_data_dir = str(tmp_path / "data")
    test_index_dir = os.path.join(test_data_dir, "main_index")
    os.makedirs(test_index_dir, exist_ok=True)
    
    # Isolate data directory using environment variables
    monkeypatch.setenv("ADVE_DATA_DIR", test_data_dir)
    
    import adve.core.clip_loader as clip_loader
    monkeypatch.setattr(clip_loader, "load_clip_cached", lambda name, device: (MockCLIP(), MagicMock()))
    
    # Mock Whisper audio transcriber
    import adve.core.audio_transcriber as audio_transcriber
    class MockWhisper:
        def __init__(self, model_name=None):
            self.whisper_available = True
        def transcribe(self, path):
            return [{"timestamp": 5.0, "text": "mock transcript"}]
    monkeypatch.setattr(audio_transcriber, "AudioTranscriber", MockWhisper)
    
    # Import FastAPI server
    from adve.api.server import app, init_users_db
    import adve.api.server as server
    
    # Reinitialize server search index and unified search engine in the new temp folder
    from adve.search.index import ADVESearchIndex
    from adve.vision.unified_search import UnifiedSearchEngine
    
    server_index = ADVESearchIndex(test_index_dir)
    monkeypatch.setattr(server, "search_index", server_index)
    
    if server.global_unified_search is not None:
        server_unified = UnifiedSearchEngine(
            visual_index=server_index,
            ocr_extractor=server.global_ocr_extractor,
            audio_indexer=server.global_audio_indexer
        )
        monkeypatch.setattr(server, "global_unified_search", server_unified)
        
    init_users_db()
    
    with TestClient(app) as client:
        yield client, server


def test_health_check(test_app):
    """Validate server health endpoint response code and structure."""
    client, _ = test_app
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_user_registration(test_app):
    """Validate user registration route contracts and key formats."""
    client, _ = test_app
    payload = {"name": "Test User", "email": "test@example.com"}
    response = client.post("/v1/register", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "api_key" in data
    assert data["api_key"].startswith("adve_live_")
    
    # Duplicate registration must return the exact same key cleanly
    response_dup = client.post("/v1/register", json=payload)
    assert response_dup.status_code == 200
    assert response_dup.json()["api_key"] == data["api_key"]


def test_api_stats(test_app):
    """Validate stats schema and initial empty properties."""
    client, _ = test_app
    response = client.get("/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "index" in data
    assert "cameras" in data
    assert "active_tasks" in data
    assert data["index"]["total_embeddings"] == 0


def test_frame_not_found(test_app):
    """Validate that requesting a frame from an unindexed/missing video returns 404."""
    client, _ = test_app
    response = client.get("/v1/frame?video_id=missing_video.mp4&frame_idx=100")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_search_empty_database(test_app):
    """Validate visual text search on an empty database returns cleanly."""
    client, _ = test_app
    payload = {"query": "person walking", "k": 5}
    response = client.post("/v1/search/text", json=payload)
    assert response.status_code == 200
    assert response.json() == []


def test_search_valid_results(test_app):
    """Validate visual text search returns matching records in correct schema formats."""
    client, server = test_app
    dim = server.search_index.dim
    
    # Generate and normalize a mock embedding
    emb = np.random.randn(dim).astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-8)
    
    # Force MockCLIP instance creation since search_index loads model lazily
    server.search_index._clip_model = MockCLIP(dim)
    # Mock the CLIP model text encoder to return the exact same embedding
    import torch
    server.search_index._clip_model.encode_text = MagicMock(
        return_value=torch.from_numpy(emb).reshape(1, -1)
    )
    
    server.search_index.add("video_a.mp4", "cam_a", 2.0, 60, emb, is_anchor=True)
    
    payload = {"query": "person walking", "k": 5}
    response = client.post("/v1/search/text", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    
    # Verify SearchResult contract properties
    r = results[0]
    assert r["video_path"].endswith("video_a.mp4")
    assert "camera_id" in r
    assert r["timestamp"] == 2.0
    assert r["frame_idx"] == 60
    assert "similarity" in r
    assert r["is_anchor"] is True


def test_invalid_inputs(test_app):
    """Validate that API returns 422 validation error for empty or malformed inputs."""
    client, _ = test_app
    response = client.post("/v1/search/text", json={})
    assert response.status_code == 422


def test_image_search_missing_file(test_app):
    """Validate that image search fails with 422 when the image payload is missing."""
    client, _ = test_app
    response = client.post("/v1/search/image")
    assert response.status_code == 422
