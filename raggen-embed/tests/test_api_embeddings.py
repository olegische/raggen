import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

from src.main import app
from src.api.embeddings import get_embedding_service, get_vector_store

client = TestClient(app)

def test_embed_text():
    """Test embedding generation for a single text."""
    text = "Sample text for embedding generation"
    
    # Mock embedding service and vector store
    with patch('src.api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('src.api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embedding.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.dimension = 384
        mock_vs.add_vectors.return_value = [1]
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_embedding_service] = lambda: mock_embed
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/embed",
                json={"text": text}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            assert len(data["embedding"]) == 384
            assert data["vector_id"] == 1
            assert data["text"] == text
            
            # Verify embedding service was called correctly
            mock_embed.get_embedding.assert_called_once_with(text)
        finally:
            app.dependency_overrides.clear()

def test_batch_embed():
    """Test batch embedding generation."""
    texts = [
        "First text for embedding",
        "Second text for embedding"
    ]
    
    # Mock embedding service and vector store
    with patch('src.api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('src.api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embeddings.return_value = np.ones((len(texts), 384))
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.dimension = 384
        mock_vs.add_vectors.return_value = [1, 2]
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_embedding_service] = lambda: mock_embed
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/embed/batch",
                json={"texts": texts}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) == len(texts)
            for i, embedding_data in enumerate(data["embeddings"]):
                assert len(embedding_data["embedding"]) == 384
                assert embedding_data["vector_id"] == i + 1
                assert embedding_data["text"] == texts[i]
            
            # Verify embedding service was called correctly
            mock_embed.get_embeddings.assert_called_once_with(texts)
        finally:
            app.dependency_overrides.clear()

def test_invalid_text():
    """Test validation of text input."""
    # Test empty text
    response = client.post(
        "/api/v1/embed",
        json={"text": ""}
    )
    assert response.status_code == 400
    
    # Test text too long
    response = client.post(
        "/api/v1/embed",
        json={"text": "x" * 513}  # More than 512 characters
    )
    assert response.status_code == 400

def test_invalid_batch():
    """Test validation of batch input."""
    # Test empty texts list
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": []}
    )
    assert response.status_code == 400
    
    # Test too many texts
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": ["text"] * 33}  # More than 32 texts
    )
    assert response.status_code == 400
    
    # Test text too long in batch
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": ["x" * 513]}  # Text longer than 512 characters
    )
    assert response.status_code == 400

def test_search_similar():
    """Test similarity search."""
    query = "Sample query text"
    
    # Mock embedding service and vector store
    with patch('src.api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('src.api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embedding.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.search.return_value = (
            np.array([[0.2, 0.3]]),  # distances
            np.array([[1, 2]])       # indices
        )
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_embedding_service] = lambda: mock_embed
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/search",
                json={
                    "text": query,
                    "k": 2
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == query
            assert len(data["results"]) == 2
            
            # Verify scores are normalized correctly
            assert data["results"][0]["score"] > data["results"][1]["score"]
            assert 0 <= data["results"][0]["score"] <= 1
            assert 0 <= data["results"][1]["score"] <= 1
            
            # Verify vector IDs match indices
            assert data["results"][0]["vector_id"] == 1
            assert data["results"][1]["vector_id"] == 2
        finally:
            app.dependency_overrides.clear()