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

@pytest.fixture
def vector_store(embedding_service) -> FAISSVectorStore:
    """Vector store fixture with sample data."""
    store = FAISSVectorStore()
    
    # Generate sample vectors
    texts = [f"Sample text {i}" for i in range(100)]
    vectors = embedding_service.get_embeddings(texts)
    
    # Train and add vectors
    store.train(vectors)
    store.add_vectors(vectors)
    
    return store

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
        assert data["vector_id"] == 1  # Check vector ID is present
        assert len(data["embedding"]) == 384  # Model dimension
        
        # Verify vector store calls
        assert store_calls["add"] == 1
        assert store_calls["train"] == 1
        
        # Test empty text
        response = client.post(
            "/api/v1/embed",
            json={"text": ""}
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "Text cannot be empty" in str(data)
        
        # Test too long text
        response = client.post(
            "/api/v1/embed",
            json={"text": "a" * 1000}
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "Text cannot be longer than 512 characters" in str(data)
        
        # Test invalid JSON
        response = client.post(
            "/api/v1/embed",
            data="invalid json"
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()

def test_embed_texts(client, monkeypatch):
    """Test batch text embedding endpoint."""
    # Mock vector store to track calls
    store_calls = {"add": 0, "train": 0}
    
    class MockVectorStore:
        def add_vectors(self, vectors):
            store_calls["add"] += 1
            return list(range(len(vectors)))  # Return mock vector IDs
            
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
        for i, emb in enumerate(data["embeddings"]):
            assert len(emb["embedding"]) == 384
            assert "vector_id" in emb
            assert emb["vector_id"] == i  # Check vector IDs match
        
        # Verify vector store calls
        assert store_calls["add"] == 1
        assert store_calls["train"] == 1
        
        # Test empty batch
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": []}
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "Texts list cannot be empty" in str(data)
        
        # Test batch with empty text
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": ["valid text", ""]}
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "Text at position 1 cannot be empty" in str(data)
        
        # Test too large batch
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": ["text"] * 50}  # Max is 32
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "list should have at most 32 items" in str(data).lower()
    finally:
        app.dependency_overrides.clear()

def test_search_similar(client, monkeypatch):
    """Test similarity search endpoint."""
    # Mock vector store to return predictable results
    class MockVectorStore:
        def __init__(self):
            self.is_trained = False
            self.index = None  # Mock index
            self.texts = []  # Store texts
            self.vectors = None  # Store vectors
            self.dimension = 384  # Model dimension
            self.n_vectors = 0  # Number of vectors
            logger.info("MockVectorStore initialized")
            
        def train(self, vectors):
            """Mock training."""
            logger.info("Training mock store with %d vectors", len(vectors))
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            self.vectors = vectors
            self.is_trained = True
            self.index = True  # Mock index initialization
            logger.info("Mock store trained successfully")
            
        def add_vectors(self, vectors):
            """Mock adding vectors."""
            logger.info("Adding %d vectors to mock store", len(vectors))
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
            self.n_vectors += len(vectors)
            logger.info("Mock store now has %d vectors", self.n_vectors)
            
        def add_texts(self, texts):
            """Mock adding texts."""
            logger.info("Adding %d texts to mock store", len(texts))
            self.texts.extend(texts)
            logger.info("Mock store now has %d texts", len(self.texts))
            
        def search(self, query_vectors, k):
            """Mock search."""
            logger.info("Searching mock store with k=%d", k)
            logger.info("Mock store state: is_trained=%s, index=%s, n_vectors=%d", 
                       self.is_trained, self.index is not None, self.n_vectors)
            if not self.is_trained or self.index is None:
                raise RuntimeError("Index must be trained before searching")
            if query_vectors.shape[1] != self.dimension:
                raise ValueError(f"Expected vectors of dimension {self.dimension}, got {query_vectors.shape[1]}")
            if k > self.n_vectors:
                k = min(k, self.n_vectors)
            n = min(k, 3)  # Return at most 3 results
            logger.info("Returning %d results", n)
            return (
                np.array([[0.1, 0.2, 0.3][:n]]),  # Distances
                np.array([[1, 2, 3][:n]])  # Indices
            )
            
        def __len__(self):
            """Get number of vectors in the store."""
            return self.n_vectors
    
    # Mock both services
    mock_store = MockVectorStore()
    mock_service = EmbeddingService()
    
    # Train the index with some sample data
    sample_texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
    sample_vectors = mock_service.get_embeddings(sample_texts)
    mock_store.train(sample_vectors)
    mock_store.add_vectors(sample_vectors)
    mock_store.add_texts(sample_texts)
    
    # Override FastAPI dependencies
    def get_mock_store():
        return mock_store
        
    def get_mock_service():
        return mock_service
    
    app.dependency_overrides[get_vector_store] = get_mock_store
    app.dependency_overrides[get_embedding_service] = get_mock_service
    
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
        for i, result in enumerate(data["results"]):
            assert 0 <= result["score"] <= 1
            assert result["vector_id"] == i + 1
        
        # Test invalid k
        response = client.post(
            "/api/v1/search",
            json={
                "text": "Query text",
                "k": 0  # Min is 1
            }
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "greater than or equal to 1" in str(data).lower()
        
        response = client.post(
            "/api/v1/search",
            json={
                "text": "Query text",
                "k": 101  # Max is 100
            }
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "less than or equal to 100" in str(data).lower()
        
        # Test empty query
        response = client.post(
            "/api/v1/search",
            json={
                "text": "",
                "k": 5
            }
        )
        assert response.status_code == 422  # Pydantic validation
        data = response.json()
        assert "Text cannot be empty" in str(data)
    finally:
        # Clean up
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