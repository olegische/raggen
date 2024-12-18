"""Tests for embeddings cache implementations."""
import numpy as np
import pytest

from core.embeddings.cache import LRUEmbeddingCache

@pytest.fixture
def cache():
    """Fixture for LRU cache."""
    return LRUEmbeddingCache(max_size=3)

def test_cache_put_get(cache):
    """Test basic cache operations."""
    text = "test text"
    embedding = np.array([1.0, 2.0, 3.0])
    
    # Put embedding in cache
    cache.put(text, embedding)
    
    # Get embedding from cache
    cached = cache.get(text)
    np.testing.assert_array_equal(cached, embedding)
    
    # Check stats
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["size"] == 1

def test_cache_miss(cache):
    """Test cache miss."""
    text = "missing text"
    
    with pytest.raises(KeyError):
        cache.get(text)
    
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["size"] == 0

def test_cache_eviction(cache):
    """Test LRU eviction."""
    # Add max_size + 1 items
    embeddings = {
        f"text{i}": np.array([float(i)]) for i in range(4)
    }
    
    for text, embedding in embeddings.items():
        cache.put(text, embedding)
    
    # First item should be evicted
    with pytest.raises(KeyError):
        cache.get("text0")
    
    # Last items should still be there
    for i in range(1, 4):
        text = f"text{i}"
        np.testing.assert_array_equal(cache.get(text), embeddings[text])
    
    stats = cache.get_stats()
    assert stats["size"] == 3  # max_size

def test_cache_update_access_order(cache):
    """Test that accessing an item moves it to the end."""
    # Add items
    for i in range(3):
        cache.put(f"text{i}", np.array([float(i)]))
    
    # Access first item
    _ = cache.get("text0")
    
    # Add new item
    cache.put("text3", np.array([3.0]))
    
    # text1 should be evicted (text0 was moved to end)
    with pytest.raises(KeyError):
        cache.get("text1")
    
    # text0 should still be there
    cache.get("text0")