"""Tests for embeddings cache functionality."""
import pytest
import numpy as np
import logging

from core.embeddings.cache.lru_cache import LRUEmbeddingCache

logger = logging.getLogger(__name__)

def test_cache_through_service(app_container, sample_text):
    """Test that cache works through embedding service."""
    # Get embedding service
    service = app_container.get_embedding_service()
    
    # First request - should miss cache
    first_embedding = service.get_embedding(sample_text)
    stats = service.get_cache_stats()
    assert stats["hits"] == 0, "Should miss cache on first request"
    assert stats["misses"] == 1, "Should have one cache miss"
    
    # Second request - should hit cache
    second_embedding = service.get_embedding(sample_text)
    stats = service.get_cache_stats()
    assert stats["hits"] == 1, "Should hit cache on second request"
    assert stats["misses"] == 1, "Misses should not increase"
    
    # Results should be identical
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Cached result should match original"
    )

def test_cache_persistence_across_service_instances(app_container, sample_text):
    """Test that cache persists when getting new service instances."""
    # Get first service instance and use it
    service1 = app_container.get_embedding_service()
    first_embedding = service1.get_embedding(sample_text)
    
    # Get second service instance
    service2 = app_container.get_embedding_service()
    assert service1 is service2, "Should return same service instance"
    
    # Use second instance - should hit cache
    second_embedding = service2.get_embedding(sample_text)
    stats = service2.get_cache_stats()
    assert stats["hits"] == 1, "Should hit cache from first instance"
    
    # Results should be identical
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Results should be identical across instances"
    )

def test_cache_size_limit(app_container):
    """Test that cache respects size limit."""
    service = app_container.get_embedding_service()
    
    # Generate texts that will produce different embeddings
    texts = [f"Test text number {i}" for i in range(20)]
    
    # Get embeddings for all texts
    embeddings = []
    for text in texts:
        embedding = service.get_embedding(text)
        embeddings.append(embedding)
    
    # Cache should have some misses due to size limit
    stats = service.get_cache_stats()
    assert stats["misses"] == len(texts), "Should miss for each new text"
    assert stats["hits"] == 0, "Should not hit cache for unique texts"
    
    # Get embeddings again - some should hit cache
    for i, text in enumerate(texts[-10:]):  # Last 10 texts
        embedding = service.get_embedding(text)
        np.testing.assert_array_equal(
            embedding,
            embeddings[i + 10],
            "Cached embeddings should match originals"
        )
    
    # Should have some cache hits now
    stats = service.get_cache_stats()
    assert stats["hits"] > 0, "Should have cache hits for recent texts"

def test_cache_with_different_texts(app_container):
    """Test cache behavior with different texts."""
    service = app_container.get_embedding_service()
    
    # Original text and similar variations
    base_text = "This is a test text for cache testing."
    similar_text = "This is a test text for cache testing!"  # Just punctuation change
    different_text = "Completely different text for testing cache."
    
    # Get embeddings
    base_embedding = service.get_embedding(base_text)
    similar_embedding = service.get_embedding(similar_text)
    different_embedding = service.get_embedding(different_text)
    
    # Check cache stats
    stats = service.get_cache_stats()
    assert stats["misses"] == 3, "Should miss cache for all different texts"
    
    # Results should be different
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            base_embedding,
            similar_embedding,
            "Even similar texts should have different embeddings"
        )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            base_embedding,
            different_embedding,
            "Different texts should have different embeddings"
        )

def test_cache_after_container_reset(app_container, sample_text):
    """Test cache behavior after container reset."""
    # Get initial service and use it
    service = app_container.get_embedding_service()
    first_embedding = service.get_embedding(sample_text)
    
    # Store settings before reset
    settings = app_container.get_settings()
    
    # Reset container
    app_container.reset()
    
    # Configure new container
    app_container.configure(settings)
    
    # Get new service
    new_service = app_container.get_embedding_service()
    assert new_service is not service, "Should be new service instance after reset"
    
    # Get embedding again
    second_embedding = new_service.get_embedding(sample_text)
    
    # Should be same embedding but from new cache
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Embeddings should be same even after reset"
    )
    
    # Should be a cache miss since it's a new cache
    stats = new_service.get_cache_stats()
    assert stats["misses"] == 1, "Should miss cache after reset"
    assert stats["hits"] == 0, "Should not have any cache hits"