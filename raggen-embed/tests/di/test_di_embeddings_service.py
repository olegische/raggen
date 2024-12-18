"""Tests for dependency injection in EmbeddingService."""
import pytest
import numpy as np
import logging

from core.embeddings import EmbeddingService
from core.embeddings.base import EmbeddingModel, EmbeddingCache
from core.embeddings.implementations.transformer_model import TransformerModel
from core.embeddings.cache.lru_cache import LRUEmbeddingCache
from tests.di.conftest import MockApplicationContainer

logger = logging.getLogger(__name__)

def test_embedding_service_with_injected_dependencies(app_container):
    """Test EmbeddingService with injected model and cache."""
    # Get dependencies from container
    settings = app_container.get_settings()
    
    # Create mock model and cache
    model = TransformerModel(lazy_init=True)
    cache = LRUEmbeddingCache(max_size=settings.batch_size * 10)
    
    # Create service with injected dependencies
    service = EmbeddingService(
        model=model,
        cache=cache,
        settings=settings
    )
    
    # Verify we got an EmbeddingService with injected dependencies
    assert isinstance(service, EmbeddingService), "Should be an EmbeddingService"
    assert service._model is model, "Model should be the injected one"
    assert service._cache is cache, "Cache should be the injected one"
    assert service._settings is settings, "Settings should be the injected one"

def test_embedding_service_caches_with_injected_cache(app_container):
    """Test that EmbeddingService correctly uses injected cache."""
    settings = app_container.get_settings()
    
    # Create mock model and cache
    model = TransformerModel(lazy_init=True)
    cache = LRUEmbeddingCache(max_size=settings.batch_size * 10)
    
    # Create service with injected dependencies
    service = EmbeddingService(
        model=model,
        cache=cache,
        settings=settings
    )
    
    # Test text to embed
    test_text = "test text"
    
    # Get embedding first time (should use model)
    first_embedding = service.get_embedding(test_text)
    
    # Get embedding second time (should use cache)
    second_embedding = service.get_embedding(test_text)
    
    # Should be same array
    np.testing.assert_array_equal(first_embedding, second_embedding)
    
    # Verify cache stats show hit
    stats = service.get_cache_stats()
    assert stats["hits"] == 1, "Cache should have been used"

def test_embedding_service_auto_creates_dependencies():
    """Test that EmbeddingService creates dependencies if none injected."""
    # Create service without injected dependencies
    service = EmbeddingService()
    
    # Should create model and cache
    assert isinstance(service._model, EmbeddingModel), "Should create model"
    assert isinstance(service._cache, EmbeddingCache), "Should create cache"
    assert service._settings is not None, "Should create settings"
    
    # Should be functional
    test_text = "test text"
    embedding = service.get_embedding(test_text)
    assert isinstance(embedding, np.ndarray), "Should return numpy array"