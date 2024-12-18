"""Tests for transformer model embeddings functionality."""
import pytest
import numpy as np
import logging

from core.embeddings.implementations.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

def test_model_basic_encoding(app_container, sample_text):
    """Test basic text encoding through embedding service."""
    # Get embedding service
    service = app_container.get_embedding_service()
    
    # Get embedding
    embedding = service.get_embedding(sample_text)
    
    # Check embedding properties
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.dtype == np.float32, "Embedding should be float32"
    assert embedding.shape == (service._settings.vector_dim,), f"Wrong embedding dimension: {embedding.shape}"
    assert np.all(np.isfinite(embedding)), "Embedding should not contain inf/nan values"

def test_model_consistency_across_service_instances(app_container, sample_text):
    """Test that model produces consistent embeddings across service instances."""
    # Get first service instance and use it
    service1 = app_container.get_embedding_service()
    first_embedding = service1.get_embedding(sample_text)
    
    # Get second service instance
    service2 = app_container.get_embedding_service()
    assert service1 is service2, "Should return same service instance"
    
    # Get embedding through second instance
    second_embedding = service2.get_embedding(sample_text)
    
    # Results should be identical
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Embeddings should be identical across service instances"
    )

def test_model_different_text_inputs(app_container):
    """Test model behavior with different text inputs."""
    service = app_container.get_embedding_service()
    
    # Test different types of text
    short_text = "Short text."
    long_text = "This is a much longer text that contains multiple sentences. " * 5
    special_chars = "Text with special chars: !@#$%^&*()_+"
    numbers = "Text with numbers: 12345 67890"
    
    # Get embeddings
    embeddings = {
        'short': service.get_embedding(short_text),
        'long': service.get_embedding(long_text),
        'special': service.get_embedding(special_chars),
        'numbers': service.get_embedding(numbers)
    }
    
    # Check all embeddings have correct shape
    for name, embedding in embeddings.items():
        assert embedding.shape == (service._settings.vector_dim,), f"Wrong shape for {name}"
        assert np.all(np.isfinite(embedding)), f"Invalid values in {name}"
    
    # Different texts should produce different embeddings
    for name1, emb1 in embeddings.items():
        for name2, emb2 in embeddings.items():
            if name1 != name2:
                with pytest.raises(AssertionError):
                    np.testing.assert_array_equal(
                        emb1,
                        emb2,
                        f"Embeddings should differ: {name1} vs {name2}"
                    )

def test_model_after_container_reset(app_container, sample_text):
    """Test model behavior after container reset."""
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
    
    # Should be same embedding even with new model instance
    np.testing.assert_array_equal(
        first_embedding,
        second_embedding,
        "Embeddings should be same even after reset"
    )

def test_model_batch_encoding(app_container):
    """Test model behavior with batch encoding."""
    service = app_container.get_embedding_service()
    
    # Create batch of texts
    texts = [
        "First test text",
        "Second test text",
        "Third test text with more content",
        "Fourth test text that is even longer than others"
    ]
    
    # Get embeddings for batch
    embeddings = service.get_embeddings(texts)
    
    # Check batch result properties
    assert isinstance(embeddings, np.ndarray), "Batch result should be numpy array"
    assert embeddings.dtype == np.float32, "Embeddings should be float32"
    assert embeddings.shape == (len(texts), service._settings.vector_dim), "Wrong batch shape"
    assert np.all(np.isfinite(embeddings)), "Embeddings should not contain inf/nan values"
    
    # Compare with individual encoding
    for i, text in enumerate(texts):
        individual_embedding = service.get_embedding(text)
        np.testing.assert_array_almost_equal(
            embeddings[i],
            individual_embedding,
            decimal=6,  # Allow small floating-point differences
            err_msg="Batch embedding should match individual encoding"
        )