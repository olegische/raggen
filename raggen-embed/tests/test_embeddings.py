import time
import numpy as np
import pytest
from core.embeddings import EmbeddingService
from core.text_processing import ParagraphProcessor, ParagraphConfig
from config.settings import Settings

settings = Settings()

@pytest.fixture
def embedding_service():
    """Fixture for embedding service with lazy initialization."""
    return EmbeddingService(lazy_init=True)

@pytest.fixture
def initialized_service():
    """Fixture for pre-initialized embedding service."""
    return EmbeddingService(lazy_init=False)

def test_model_initialization(embedding_service, initialized_service):
    """Test model initialization and lazy loading."""
    # Lazy initialization
    assert embedding_service._model is None
    
    # First access should initialize the model
    _ = embedding_service.model
    assert embedding_service._model is not None
    assert embedding_service.load_time > 0
    
    # Immediate initialization
    assert initialized_service._model is not None
    assert initialized_service.load_time > 0
    
    # Model should match configured name
    assert embedding_service.model.get_sentence_embedding_dimension() == settings.vector_dim

def test_get_embedding(initialized_service):
    """Test single text embedding."""
    text = "This is a test text"
    embedding = initialized_service.get_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (settings.vector_dim,)
    
    # Test vector properties
    assert not np.any(np.isnan(embedding))  # No NaN values
    assert not np.any(np.isinf(embedding))  # No infinite values
    assert -1 <= np.max(embedding) <= 1  # Values should be normalized

def test_get_embeddings(initialized_service):
    """Test batch text embedding."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = initialized_service.get_embeddings(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), settings.vector_dim)
    
    # Test vector properties
    assert not np.any(np.isnan(embeddings))  # No NaN values
    assert not np.any(np.isinf(embeddings))  # No infinite values
    assert -1 <= np.max(embeddings) <= 1  # Values should be normalized

def test_embedding_caching(initialized_service):
    """Test that embeddings are properly cached."""
    text = "This is a text that should be cached"
    
    # First call should be a cache miss
    embedding1 = initialized_service.get_embedding(text)
    stats1 = initialized_service.get_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0
    
    # Second call with same text should be a cache hit
    embedding2 = initialized_service.get_embedding(text)
    stats2 = initialized_service.get_cache_stats()
    assert stats2["misses"] == 1
    assert stats2["hits"] == 1
    
    # Embeddings should be identical
    np.testing.assert_array_equal(embedding1, embedding2)

def test_error_handling(initialized_service):
    """Test error handling for invalid inputs."""
    # Test empty text
    with pytest.raises(ValueError, match="Empty text at position 0"):
        initialized_service.get_embedding("")
    
    # Test empty list
    with pytest.raises(ValueError, match="Empty text list provided"):
        initialized_service.get_embeddings([])
    
    # Test list with empty text
    with pytest.raises(ValueError, match="Empty text at position"):
        initialized_service.get_embeddings(["", ""])
    
    # Test text exceeding max length
    long_text = "a" * (settings.max_text_length + 1)
    with pytest.raises(ValueError, match="exceeds maximum length"):
        initialized_service.get_embedding(long_text)

def test_performance(initialized_service):
    """Test embedding generation performance."""
    # Test single text performance
    text = "This is a test text"
    start_time = time.time()
    _ = initialized_service.get_embedding(text)
    single_time = time.time() - start_time
    
    # Test batch performance
    texts = [text] * settings.batch_size
    start_time = time.time()
    _ = initialized_service.get_embeddings(texts)
    batch_time = time.time() - start_time
    
    # Batch processing should be more efficient per text
    assert batch_time < single_time * settings.batch_size
    
    # Log performance metrics
    print(f"\nPerformance metrics:")
    print(f"Single text time: {single_time:.3f}s")
    print(f"Batch processing time ({settings.batch_size} texts): {batch_time:.3f}s")
    print(f"Average time per text in batch: {batch_time/settings.batch_size:.3f}s")

def test_paragraph_processing_single(initialized_service):
    """Test paragraph processing for single text."""
    # Create a long text with multiple paragraphs
    text = "First paragraph with some content. " * 5 + "\n\n" + \
           "Second paragraph with different content. " * 5 + "\n\n" + \
           "Third paragraph for testing. " * 5
    
    # Get embedding without paragraph processing
    regular_embedding = initialized_service.get_embedding(text)
    
    # Get embedding with paragraph processing
    paragraph_embedding = initialized_service.get_embedding(text, use_paragraphs=True)
    
    # Verify embeddings are different
    assert not np.array_equal(regular_embedding, paragraph_embedding)
    
    # Verify embedding dimensions
    assert regular_embedding.shape == (settings.vector_dim,)
    assert paragraph_embedding.shape == (settings.vector_dim,)
    
    # Verify embedding properties
    assert not np.any(np.isnan(paragraph_embedding))
    assert not np.any(np.isinf(paragraph_embedding))
    assert -1 <= np.max(paragraph_embedding) <= 1

def test_paragraph_processing_batch(initialized_service):
    """Test paragraph processing for batch texts."""
    # Create multiple texts with paragraphs
    texts = [
        "First text paragraph one. " * 3 + "\n\n" + "First text paragraph two. " * 3,
        "Second text paragraph one. " * 3 + "\n\n" + "Second text paragraph two. " * 3
    ]
    
    # Get embeddings without paragraph processing
    regular_embeddings = initialized_service.get_embeddings(texts)
    
    # Get embeddings with paragraph processing
    paragraph_embeddings = initialized_service.get_embeddings(texts, use_paragraphs=True)
    
    # Verify shapes
    assert regular_embeddings.shape == (len(texts), settings.vector_dim)
    assert paragraph_embeddings.shape == (len(texts), settings.vector_dim)
    
    # Verify embeddings are different
    assert not np.array_equal(regular_embeddings, paragraph_embeddings)
    
    # Verify embedding properties
    assert not np.any(np.isnan(paragraph_embeddings))
    assert not np.any(np.isinf(paragraph_embeddings))
    assert -1 <= np.max(paragraph_embeddings) <= 1

def test_paragraph_processor_config(initialized_service):
    """Test paragraph processor configuration."""
    text = "Test paragraph one. " * 5 + "\n\n" + "Test paragraph two. " * 5
    
    # Configure custom paragraph processor
    custom_config = ParagraphConfig(
        max_length=200,
        min_length=50,
        overlap=30,
        preserve_sentences=True
    )
    initialized_service._paragraph_processor = ParagraphProcessor(custom_config)
    
    # Get embedding with custom paragraph processing
    embedding = initialized_service.get_embedding(text, use_paragraphs=True)
    
    # Verify embedding
    assert embedding.shape == (settings.vector_dim,)
    assert not np.any(np.isnan(embedding))
    assert not np.any(np.isinf(embedding))
    assert -1 <= np.max(embedding) <= 1

def test_paragraph_merge_strategies(initialized_service):
    """Test different paragraph merge strategies."""
    text = "First paragraph for testing. " * 5 + "\n\n" + \
           "Second paragraph for testing. " * 5
    
    # Test mean strategy
    settings.embedding_merge_strategy = "mean"
    mean_embedding = initialized_service.get_embedding(text, use_paragraphs=True)
    
    # Test weighted strategy
    settings.embedding_merge_strategy = "weighted"
    weighted_embedding = initialized_service.get_embedding(text, use_paragraphs=True)
    
    # Verify embeddings are different
    assert not np.array_equal(mean_embedding, weighted_embedding)
    
    # Verify embedding properties
    for embedding in [mean_embedding, weighted_embedding]:
        assert embedding.shape == (settings.vector_dim,)
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))
        assert -1 <= np.max(embedding) <= 1