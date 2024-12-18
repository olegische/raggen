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

def test_embedding_service_singleton_from_container(app_container):
    """Test that EmbeddingService is a singleton in ApplicationContainer."""
    # Get service multiple times
    service1 = app_container.get_embedding_service()
    service2 = app_container.get_embedding_service()
    
    # Should be same instance
    assert service1 is service2, "Should return same EmbeddingService instance"
    
    # Internal dependencies should also be same instances
    assert service1._model is service2._model, "Should share same model instance"
    assert service1._cache is service2._cache, "Should share same cache instance"

def test_embedding_service_maintains_state_across_uses(app_container):
    """Test that EmbeddingService maintains state (cache) across different uses."""
    service = app_container.get_embedding_service()
    logger.info("Got initial service instance (id: %x)", id(service))
    
    # First use
    test_text = "test text for state persistence"
    first_embedding = service.get_embedding(test_text)
    logger.info("Generated first embedding, shape: %s", first_embedding.shape)
    
    # Get service again and use
    service_again = app_container.get_embedding_service()
    logger.info("Got service again (id: %x)", id(service_again))
    second_embedding = service_again.get_embedding(test_text)
    logger.info("Generated second embedding, shape: %s", second_embedding.shape)
    
    # Should use cached result
    np.testing.assert_array_equal(first_embedding, second_embedding)
    stats = service_again.get_cache_stats()
    logger.info("Cache stats after reuse: %s", stats)
    assert stats["hits"] == 1, "Should use shared cache across service instances"

def test_embedding_service_state_persists_container_reset(app_container):
    """Test that EmbeddingService state persists when container is reset."""
    # Get initial service and use it
    service = app_container.get_embedding_service()
    test_text = "test text for persistence"
    first_embedding = service.get_embedding(test_text)
    
    # Reset container
    app_container.reset()
    
    # Configure new container with same settings
    settings = app_container.get_settings()
    app_container.configure(settings)
    
    # Get new service instance and use it
    new_service = app_container.get_embedding_service()
    second_embedding = new_service.get_embedding(test_text)
    
    # Should get same result
    np.testing.assert_array_equal(first_embedding, second_embedding)
    stats = new_service.get_cache_stats()
    assert stats["hits"] == 1, "Cache should persist across container resets"

def test_embedding_service_integration_with_text_splitter(app_container):
    """Test EmbeddingService integration with TextSplitterService."""
    from container.request import RequestContainer
    
    # Get text splitter service which uses embedding service
    text_splitter = RequestContainer.get_text_splitter_service()
    
    # Verify it uses singleton embedding service
    assert text_splitter.embedding_service is app_container.get_embedding_service(), \
        "TextSplitterService should use singleton EmbeddingService"
    
    # Test functionality
    test_text = "test text for integration"
    embeddings = text_splitter.get_embeddings(test_text)
    assert isinstance(embeddings, np.ndarray), "Should generate embeddings through service chain"

def test_embedding_service_requires_container_configuration():
    """Test that EmbeddingService requires container to be configured."""
    # Create clean container
    from tests.di.conftest import MockApplicationContainer
    container = MockApplicationContainer
    
    # Reset to clear any configuration
    container.reset()
    
    # Should raise error when getting service
    with pytest.raises(RuntimeError, match="Container not configured"):
        container.get_embedding_service()

def test_embedding_service_cleanup_on_container_reset(app_container):
    """Test that EmbeddingService is properly cleaned up on container reset."""
    # Get initial service
    service = app_container.get_embedding_service()
    logger.info("Got initial service (id: %x)", id(service))
    
    # Store references to internal components
    model = service._model
    cache = service._cache
    logger.info("Initial components:")
    logger.info("  Model (id: %x)", id(model))
    logger.info("  Cache (id: %x)", id(cache))
    
    # Use service to verify it's working
    test_text = "test text before reset"
    embedding = service.get_embedding(test_text)
    logger.info("Generated embedding before reset, shape: %s", embedding.shape)
    
    # Reset container
    logger.info("Resetting container")
    app_container.reset()
    
    # Service should be cleared
    logger.info("Checking container state after reset")
    assert app_container._embedding_service is None, "Service should be cleared on reset"
    
    # Configure new container
    logger.info("Configuring new container")
    settings = app_container.get_settings()
    app_container.configure(settings)
    
    # Get new service
    new_service = app_container.get_embedding_service()
    logger.info("Got new service after reset (id: %x)", id(new_service))
    logger.info("New components:")
    logger.info("  Model (id: %x)", id(new_service._model))
    logger.info("  Cache (id: %x)", id(new_service._cache))
    
    # Should have new instances
    assert new_service._model is not model, "Should create new model instance after reset"
    assert new_service._cache is not cache, "Should create new cache instance after reset"
    
    # Verify new service is functional
    new_embedding = new_service.get_embedding(test_text)
    logger.info("Generated embedding after reset, shape: %s", new_embedding.shape)

def test_embedding_service_consistency_across_resets(app_container):
    """Test that EmbeddingService produces consistent results across container resets."""
    # Get initial service
    service = app_container.get_embedding_service()
    logger.info("Got initial service (id: %x)", id(service))
    
    # Generate embeddings with initial service
    test_texts = [
        "first test text",
        "second test text with more words",
        "third test text that is even longer than others"
    ]
    logger.info("Generating embeddings for %d texts", len(test_texts))
    
    original_embeddings = []
    for text in test_texts:
        embedding = service.get_embedding(text)
        original_embeddings.append(embedding)
        logger.info("Generated embedding for text length %d, shape: %s", 
                   len(text), embedding.shape)
    
    # Reset container
    logger.info("Resetting container")
    app_container.reset()
    
    # Configure new container
    logger.info("Configuring new container")
    settings = app_container.get_settings()
    app_container.configure(settings)
    
    # Get new service
    new_service = app_container.get_embedding_service()
    logger.info("Got new service after reset (id: %x)", id(new_service))
    
    # Generate embeddings with new service
    logger.info("Generating embeddings with new service")
    for i, text in enumerate(test_texts):
        new_embedding = new_service.get_embedding(text)
        logger.info("Generated new embedding for text length %d, shape: %s",
                   len(text), new_embedding.shape)
        
        # Results should be identical
        np.testing.assert_array_equal(
            original_embeddings[i],
            new_embedding,
            f"Embeddings should be identical for text: {text}"
        )
    
    logger.info("All embeddings verified to be consistent across container reset")