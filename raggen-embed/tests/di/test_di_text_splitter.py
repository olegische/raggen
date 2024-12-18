"""Tests for dependency injection in TextSplitterService."""
import pytest
import numpy as np
import logging

from core.text_splitting.service import TextSplitterService
from core.text_splitting.strategies import SlidingWindowStrategy
from tests.di.conftest import MockApplicationContainer, MockRequestContainer

logger = logging.getLogger(__name__)

def test_text_splitter_with_injected_embedding_service(app_container):
    """Test TextSplitterService with injected dependencies."""
    # Get dependencies from container
    settings = app_container.get_settings()
    embedding_service = app_container.get_embedding_service()
    split_strategy = app_container.get_text_split_strategy()
    
    # Create service with injected dependencies
    service = TextSplitterService(
        embedding_service=embedding_service,
        split_strategy=split_strategy,
        settings=settings
    )
    
    # Verify we got a TextSplitterService with injected dependencies
    assert isinstance(service, TextSplitterService), "Should be a TextSplitterService"
    assert service.embedding_service is embedding_service, "EmbeddingService should be the injected one"
    assert service.split_strategy is split_strategy, "SplitStrategy should be the injected one"
    assert service.settings is settings, "Settings should be the injected one"

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

def test_text_splitter_requires_embedding_service():
    """Test that TextSplitterService requires EmbeddingService."""
    with pytest.raises(ValueError, match="Embedding service must be provided"):
        TextSplitterService(embedding_service=None)