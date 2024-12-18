"""Tests for dependency injection in TextSplitterService."""
import pytest
import numpy as np
import logging

from core.text_splitting.service import TextSplitterService
from core.text_splitting.strategies import SlidingWindowStrategy
from tests.di.conftest import MockApplicationContainer, MockRequestContainer

logger = logging.getLogger(__name__)

def test_text_splitter_service_creation(app_container, request_container):
    """Test TextSplitterService creation through request container."""
    # Get service from request container
    service = request_container.get_text_splitter_service()
    
    # Verify service was created with correct dependencies
    assert isinstance(service, TextSplitterService), \
        "Should be a TextSplitterService"
    assert service.embedding_service is app_container.get_embedding_service(), \
        "Should use singleton EmbeddingService from ApplicationContainer"
    assert isinstance(service.split_strategy, SlidingWindowStrategy), \
        "Should use SlidingWindowStrategy by default"
    assert service.settings is app_container.get_settings(), \
        "Should use singleton Settings from ApplicationContainer"

def test_text_splitter_uses_application_embedding_service(app_container, request_container):
    """Test that TextSplitterService uses singleton EmbeddingService from container."""
    # Get the singleton EmbeddingService
    app_embedding_service = app_container.get_embedding_service()
    
    # Create text splitter through request container
    text_splitter = request_container.get_text_splitter_service()
    
    # Verify it uses the same EmbeddingService instance
    assert text_splitter.embedding_service is app_embedding_service, \
        "Should use singleton EmbeddingService from ApplicationContainer"

def test_text_splitter_with_embedding_service_integration(app_container, request_container):
    """Test TextSplitterService integration with EmbeddingService."""
    # Create text splitter through request container
    text_splitter = request_container.get_text_splitter_service()
    
    # Test text (long enough to meet min_length requirement)
    test_text = """
    This is a longer test text that will be split and embedded by our service.
    It needs to be at least 100 characters long to meet the minimum length requirement
    set in our application settings. This text should be sufficient for testing the
    text splitting and embedding functionality.
    """
    
    # Get embeddings through text splitter
    embeddings = text_splitter.get_embeddings(test_text)
    
    # Verify embeddings were generated
    assert isinstance(embeddings, np.ndarray), "Should return numpy array"
    assert len(embeddings.shape) == 2, "Should be 2D array (chunks × embedding_dim)"
    assert embeddings.shape[1] == app_container.get_settings().vector_dim, \
        "Embedding dimension should match settings"

def test_text_splitter_creates_new_instance_per_request(request_container):
    """Test that TextSplitterService creates new instance for each request."""
    # Get two instances
    service1 = request_container.get_text_splitter_service()
    service2 = request_container.get_text_splitter_service()
    
    # Verify they are different instances
    assert service1 is not service2, \
        "Should create new TextSplitterService instance for each request"
    
    # But they share the same singleton dependencies
    assert service1.embedding_service is service2.embedding_service, \
        "Should use same singleton EmbeddingService"
    assert service1.settings is service2.settings, \
        "Should use same singleton Settings"

def test_text_splitter_requires_embedding_service():
    """Test that TextSplitterService requires EmbeddingService."""
    with pytest.raises(ValueError, match="Embedding service must be provided"):
        TextSplitterService(embedding_service=None)