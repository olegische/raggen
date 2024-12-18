"""Tests for embedding model implementations."""
import time
import numpy as np
import pytest

from core.embeddings.implementations import TransformerModel
from config.settings import Settings

settings = Settings()

@pytest.fixture
def model():
    """Fixture for transformer model with lazy initialization."""
    return TransformerModel(lazy_init=True)

@pytest.fixture
def initialized_model():
    """Fixture for pre-initialized transformer model."""
    return TransformerModel(lazy_init=False)

def test_model_initialization(model, initialized_model):
    """Test model initialization and lazy loading."""
    # Lazy initialization
    assert model._model is None
    
    # First access should initialize the model
    _ = model.model
    assert model._model is not None
    assert model.load_time > 0
    
    # Immediate initialization
    assert initialized_model._model is not None
    assert initialized_model.load_time > 0
    
    # Model should match configured dimension
    assert model.model.get_sentence_embedding_dimension() == settings.vector_dim

def test_encode_single(initialized_model):
    """Test encoding single text."""
    text = "This is a test text"
    embeddings = initialized_model.encode([text])
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, settings.vector_dim)
    
    # Test vector properties
    assert not np.any(np.isnan(embeddings))  # No NaN values
    assert not np.any(np.isinf(embeddings))  # No infinite values
    assert -1 <= np.max(embeddings) <= 1  # Values should be normalized

def test_encode_batch(initialized_model):
    """Test batch encoding."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = initialized_model.encode(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), settings.vector_dim)
    
    # Test vector properties
    assert not np.any(np.isnan(embeddings))  # No NaN values
    assert not np.any(np.isinf(embeddings))  # No infinite values
    assert -1 <= np.max(embeddings) <= 1  # Values should be normalized

def test_encode_performance(initialized_model):
    """Test encoding performance."""
    # Test single text performance
    text = "This is a test text"
    start_time = time.time()
    _ = initialized_model.encode([text])
    single_time = time.time() - start_time
    
    # Test batch performance
    texts = [text] * settings.batch_size
    start_time = time.time()
    _ = initialized_model.encode(texts)
    batch_time = time.time() - start_time
    
    # Batch processing should be more efficient per text
    assert batch_time < single_time * settings.batch_size
    
    # Log performance metrics
    print(f"\nPerformance metrics:")
    print(f"Single text time: {single_time:.3f}s")
    print(f"Batch processing time ({settings.batch_size} texts): {batch_time:.3f}s")
    print(f"Average time per text in batch: {batch_time/settings.batch_size:.3f}s")