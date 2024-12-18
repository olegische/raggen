"""Tests for embedding service functionality."""
import pytest
import numpy as np
import logging

from core.embeddings import DefaultEmbeddingService

logger = logging.getLogger(__name__)

def test_service_initialization(app_container):
    """Test embedding service initialization."""
    service = app_container.get_embedding_service()
    
    # Check service properties
    assert isinstance(service, DefaultEmbeddingService)
    assert service._settings is not None
    assert service._model is not None
    assert service._cache is not None

def test_service_with_empty_input(app_container):
    """Test service behavior with empty input."""
    service = app_container.get_embedding_service()
    
    # Empty text
    with pytest.raises(ValueError, match="Empty text list provided"):
        service.get_embeddings([])
    
    # None text
    with pytest.raises(TypeError, match="Expected string input, got <class 'NoneType'>"):
        service.get_embedding(None)
    
    # Empty string
    with pytest.raises(ValueError, match="Empty text at position 0"):
        service.get_embedding("")
    
    # Whitespace string
    with pytest.raises(ValueError, match="Empty text at position 0"):
        service.get_embedding("   ")

def test_service_text_length_limit(app_container):
    """Test service text length limit handling."""
    service = app_container.get_embedding_service()
    max_length = service._settings.max_text_length
    
    # Text at limit should work
    text_at_limit = "x" * max_length
    embedding = service.get_embedding(text_at_limit)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (service._settings.vector_dim,)
    
    # Text over limit should fail
    text_over_limit = "x" * (max_length + 1)
    with pytest.raises(ValueError, match=f"Text at position 0 exceeds maximum length of {max_length}"):
        service.get_embedding(text_over_limit)

def test_service_batch_processing(app_container):
    """Test service batch processing functionality."""
    service = app_container.get_embedding_service()
    
    # Create texts of different lengths
    texts = [
        "Short text",
        "Medium length text for testing",
        "A longer text that contains multiple words and tests batch processing"
    ]
    
    # Get embeddings in batch
    embeddings = service.get_embeddings(texts)
    
    # Check batch result
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), service._settings.vector_dim)
    assert embeddings.dtype == np.float32
    
    # Compare with individual processing
    for i, text in enumerate(texts):
        individual = service.get_embedding(text)
        np.testing.assert_array_almost_equal(
            embeddings[i],
            individual,
            decimal=6,
            err_msg=f"Batch result differs for text {i}"
        )

def test_service_cache_integration(app_container):
    """Test service integration with cache."""
    service = app_container.get_embedding_service()
    
    # First request - should miss cache
    text = "Test text for cache integration"
    first_embedding = service.get_embedding(text)
    stats = service.get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    
    # Second request - should hit cache
    second_embedding = service.get_embedding(text)
    stats = service.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    
    # Results should be identical
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Cached result should match original"
    )

def test_service_error_handling(app_container):
    """Test service error handling."""
    service = app_container.get_embedding_service()
    
    # Test with invalid input types
    with pytest.raises(TypeError, match="Expected string input, got <class 'int'>"):
        service.get_embedding(123)  # Non-string input
    
    with pytest.raises(TypeError, match="Expected string input, got <class 'list'>"):
        service.get_embedding([])  # List instead of string
    
    with pytest.raises(TypeError, match="Expected list of texts, got <class 'str'>"):
        service.get_embeddings("text")  # String instead of list - raises TypeError with type mismatch message
    
    # Test with invalid batch input
    with pytest.raises(TypeError, match="Expected string at position 1, got <class 'NoneType'>"):
        service.get_embeddings(["valid text", None, "valid text"])  # None in batch
    
    with pytest.raises(TypeError, match="Expected string at position 1, got <class 'int'>"):
        service.get_embeddings(["valid text", 123, "valid text"])  # Non-string in batch
    
    with pytest.raises(ValueError, match="Empty text at position 1"):
        service.get_embeddings(["valid text", "", "valid text"])  # Empty string in batch

def test_service_dimension_consistency(app_container):
    """Test service output dimension consistency."""
    service = app_container.get_embedding_service()
    expected_dim = service._settings.vector_dim
    
    # Test with various inputs
    texts = [
        "Very short.",
        "A bit longer text.",
        "An even longer text with more content.",
        "A much longer text that contains multiple sentences and tests dimension consistency."
    ]
    
    # Check individual processing
    for text in texts:
        embedding = service.get_embedding(text)
        assert embedding.shape == (expected_dim,), f"Wrong dimension for text: {text}"
    
    # Check batch processing
    batch_embeddings = service.get_embeddings(texts)
    assert batch_embeddings.shape == (len(texts), expected_dim), "Wrong batch dimensions"