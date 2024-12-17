"""Tests for embedding service."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.embeddings import (
    EmbeddingService,
    EmbeddingModel,
    EmbeddingCache,
    DefaultEmbeddingService
)
from config.settings import Settings

@pytest.fixture
def mock_model():
    """Fixture for mock embedding model."""
    model = MagicMock(spec=EmbeddingModel)
    # Setup encode to return valid embeddings
    def mock_encode(texts, convert_to_numpy=True):
        return np.random.randn(len(texts), 384).astype(np.float32)
    model.encode.side_effect = mock_encode
    return model

@pytest.fixture
def mock_cache():
    """Fixture for mock embedding cache."""
    cache = MagicMock(spec=EmbeddingCache)
    cache.get_stats.return_value = {"hits": 0, "misses": 0, "total": 0, "size": 0}
    return cache

@pytest.fixture
def service(mock_model, mock_cache):
    """Fixture for embedding service with mocks."""
    return DefaultEmbeddingService(
        model=mock_model,
        cache=mock_cache,
        settings=Settings()
    )

def test_get_embedding(service, mock_model, mock_cache):
    """Test single text embedding with cache."""
    text = "test text"
    expected_embedding = np.array([1.0, 2.0, 3.0])
    
    # Setup cache miss then hit
    mock_cache.get.side_effect = [KeyError, expected_embedding]
    mock_model.encode.return_value = expected_embedding.reshape(1, -1)
    
    # First call - cache miss
    embedding1 = service.get_embedding(text)
    np.testing.assert_array_equal(embedding1, expected_embedding)
    mock_cache.get.assert_called_with(text)
    mock_model.encode.assert_called_once()
    mock_cache.put.assert_called_once()
    
    # Second call - cache hit
    embedding2 = service.get_embedding(text)
    np.testing.assert_array_equal(embedding2, expected_embedding)
    assert mock_model.encode.call_count == 1  # No additional calls

def test_get_embeddings(service, mock_model):
    """Test batch text embedding."""
    texts = ["first text", "second text"]
    expected_embeddings = np.random.randn(2, 384).astype(np.float32)
    mock_model.encode.return_value = expected_embeddings
    
    embeddings = service.get_embeddings(texts)
    np.testing.assert_array_equal(embeddings, expected_embeddings)
    mock_model.encode.assert_called_once_with(texts)

def test_input_validation(service):
    """Test input validation."""
    settings = Settings()
    
    # Test empty text
    with pytest.raises(ValueError, match="Empty text at position 0"):
        service.get_embedding("")
    
    # Test empty list
    with pytest.raises(ValueError, match="Empty text list provided"):
        service.get_embeddings([])
    
    # Test list with empty text
    with pytest.raises(ValueError, match="Empty text at position"):
        service.get_embeddings(["", ""])
    
    # Test text exceeding max length
    long_text = "a" * (settings.max_text_length + 1)
    with pytest.raises(ValueError, match="exceeds maximum length"):
        service.get_embedding(long_text)

def test_cache_stats(service, mock_cache):
    """Test cache statistics."""
    expected_stats = {
        "hits": 5,
        "misses": 3,
        "total": 8,
        "size": 3
    }
    mock_cache.get_stats.return_value = expected_stats
    
    stats = service.get_cache_stats()
    assert stats == expected_stats
    mock_cache.get_stats.assert_called_once()

def test_real_service_integration():
    """Integration test with real components."""
    service = DefaultEmbeddingService()  # Use real implementations
    
    # Test single embedding
    text = "This is a test text"
    embedding = service.get_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)  # Default dimension
    
    # Test batch embeddings
    texts = ["First text", "Second text"]
    embeddings = service.get_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    
    # Test caching
    _ = service.get_embedding(text)  # Should be cached
    stats = service.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1