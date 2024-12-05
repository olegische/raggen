import json
from typing import Generator
import pytest
from fastapi.testclient import TestClient
import numpy as np
import logging

from main import app
from core.embeddings import EmbeddingService
from core.vector_store.faiss_store import FAISSVectorStore
from api.embeddings import get_embedding_service, get_vector_store, _embedding_service, _vector_store

logger = logging.getLogger(__name__)

@pytest.fixture
def client() -> Generator:
    """Test client fixture."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Embedding service fixture."""
    return EmbeddingService()

@pytest.fixture(autouse=True)
def reset_global_services():
    """Reset global services before each test."""
    global _embedding_service, _vector_store
    _embedding_service = None
    _vector_store = None
    yield
    _embedding_service = None
    _vector_store = None

def test_embed_text(client, monkeypatch):
    """Test single text embedding endpoint."""
    # Mock vector store to track calls
    store_calls = {"add": 0, "train": 0}
    
    class MockVectorStore:
        def add_vectors(self, vectors):
            store_calls["add"] += 1
            return [1]  # Return mock vector ID
            
        def train(self):
            store_calls["train"] += 1
    
    # Override FastAPI dependency
    app.dependency_overrides[get_vector_store] = lambda: MockVectorStore()
    
    try:
        # Test successful embedding
        response = client.post(
            "/api/v1/embed",
            json={"text": "This is a test text"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "text" in data
        assert "vector_id" in data
        assert data["vector_id"] == 1
        assert len(data["embedding"]) == 384
        
        # Verify vector store operations
        assert store_calls["add"] == 1
        assert store_calls["train"] == 1
        
        # Test empty text
        response = client.post(
            "/api/v1/embed",
            json={"text": ""}
        )
        assert response.status_code == 422
        data = response.json()
        assert "String should have at least 1 character" in str(data)
        
        # Test too long text
        response = client.post(
            "/api/v1/embed",
            json={"text": "a" * 1000}
        )
        assert response.status_code == 422
        data = response.json()
        assert "String should have at most 512 characters" in str(data)
    finally:
        app.dependency_overrides.clear()

def test_embed_texts(client, monkeypatch):
    """Test batch text embedding endpoint."""
    # Mock vector store to track calls
    store_calls = {"add": 0, "train": 0}
    
    class MockVectorStore:
        def add_vectors(self, vectors):
            store_calls["add"] += 1
            return list(range(len(vectors)))  # Return sequential vector IDs
            
        def train(self):
            store_calls["train"] += 1
    
    # Override FastAPI dependency
    app.dependency_overrides[get_vector_store] = lambda: MockVectorStore()
    
    try:
        # Test successful batch embedding
        texts = ["First text", "Second text", "Third text"]
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": texts}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == len(texts)
        
        # Check each embedding response
        for i, emb in enumerate(data["embeddings"]):
            assert len(emb["embedding"]) == 384
            assert "vector_id" in emb
            assert emb["vector_id"] == i
            assert emb["text"] == texts[i]
        
        # Verify vector store operations
        assert store_calls["add"] == 1
        assert store_calls["train"] == 1
        
        # Test empty batch
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": []}
        )
        assert response.status_code == 422
        data = response.json()
        assert "Texts list cannot be empty" in str(data)
        
        # Test batch with empty text
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": ["valid text", ""]}
        )
        assert response.status_code == 422
        data = response.json()
        assert "Text at position 1 cannot be empty" in str(data)
        
        # Test too large batch
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": ["text"] * 50}  # Max is 32
        )
        assert response.status_code == 422
        data = response.json()
        assert "list should have at most 32 items" in str(data).lower()
    finally:
        app.dependency_overrides.clear()

def test_search_similar(client, monkeypatch):
    """Test similarity search endpoint."""
    # Mock vector store
    class MockVectorStore:
        def __init__(self):
            self.is_trained = True
            self.index = True
            self.dimension = 384
            self.n_vectors = 3
            
        def search(self, query_vectors, k):
            assert query_vectors.shape[1] == self.dimension
            n = min(k, self.n_vectors)
            return (
                np.array([[0.1, 0.2, 0.3][:n]]),  # Distances
                np.array([[1, 2, 3][:n]])  # Indices
            )
    
    # Override FastAPI dependencies
    app.dependency_overrides[get_vector_store] = lambda: MockVectorStore()
    
    try:
        # Test successful search
        response = client.post(
            "/api/v1/search",
            json={
                "text": "Query text",
                "k": 3
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Query text"
        assert len(data["results"]) == 3
        
        # Check search results
        for i, result in enumerate(data["results"]):
            assert 0 <= result["score"] <= 1
            assert result["vector_id"] == i + 1
            assert isinstance(result["text"], str)
        
        # Test invalid k
        response = client.post(
            "/api/v1/search",
            json={
                "text": "Query text",
                "k": 0
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "greater than or equal to 1" in str(data).lower()
        
        # Test empty query
        response = client.post(
            "/api/v1/search",
            json={
                "text": "",
                "k": 5
            }
        )
        assert response.status_code == 422
        data = response.json()
        assert "String should have at least 1 character" in str(data)
    finally:
        app.dependency_overrides.clear()

def test_error_handling(client):
    """Test API error handling."""
    # Test invalid endpoint
    response = client.post("/api/v1/invalid")
    assert response.status_code == 404
    
    # Test method not allowed
    response = client.get("/api/v1/embed")
    assert response.status_code == 405
    
    # Test invalid content type
    response = client.post(
        "/api/v1/embed",
        data="plain text",
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422