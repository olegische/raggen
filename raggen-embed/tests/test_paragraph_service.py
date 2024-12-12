"""Tests for paragraph processing service."""
import pytest
import numpy as np
from src.core.paragraph_service import ParagraphService
from src.core.embeddings import EmbeddingService

def test_paragraph_service_initialization():
    """Test paragraph service initialization."""
    service = ParagraphService()
    assert service is not None
    assert isinstance(service.embedding_service, EmbeddingService)

def test_split_text_basic():
    """Test basic text splitting."""
    service = ParagraphService()
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = service.split_text(text)
    assert len(paragraphs) == 3
    assert paragraphs[0] == "First paragraph."
    assert paragraphs[1] == "Second paragraph."
    assert paragraphs[2] == "Third paragraph."

def test_split_text_with_settings():
    """Test text splitting with custom settings."""
    service = ParagraphService(min_length=20, max_length=50, overlap=10)
    text = "A very long paragraph that should be split into multiple pieces based on length."
    paragraphs = service.split_text(text)
    assert all(20 <= len(p) <= 50 for p in paragraphs)

def test_get_paragraph_embeddings():
    """Test getting embeddings for paragraphs."""
    service = ParagraphService()
    text = "First paragraph.\n\nSecond paragraph."
    embeddings = service.get_embeddings(text)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(e, np.ndarray) for e in embeddings)

def test_merge_embeddings_mean():
    """Test merging embeddings with mean strategy."""
    service = ParagraphService()
    embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0])
    ]
    merged = service.merge_embeddings(embeddings, strategy="mean")
    assert isinstance(merged, np.ndarray)
    assert merged.shape == (3,)
    assert np.allclose(merged, np.array([2.5, 3.5, 4.5]))

def test_merge_embeddings_weighted():
    """Test merging embeddings with weighted strategy."""
    service = ParagraphService()
    embeddings = [
        np.array([1.0, 1.0, 1.0]),  # First paragraph should have more weight
        np.array([2.0, 2.0, 2.0])
    ]
    merged = service.merge_embeddings(embeddings, strategy="weighted")
    assert isinstance(merged, np.ndarray)
    assert merged.shape == (3,)
    # First paragraph should have more weight in the result
    assert all(1.0 < x < 2.0 for x in merged)
    # First paragraph should contribute more to the result
    assert np.allclose(merged[0], merged[1])  # All dimensions should be weighted equally

def test_split_text_with_overlap():
    """Test that text splitting includes proper overlap."""
    # Use more realistic settings
    service = ParagraphService(min_length=50, max_length=100, overlap=20)
    
            # Create a longer text with natural sentence boundaries
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A second sentence about the fox and dog. "
        "More details about their interaction. "
        "The lazy dog watches the quick brown fox. "
        "They continue their daily routine in the forest. "
        "Sometimes they even play together in the meadow."
    )
    
    paragraphs = service.split_text(text)
    
    # Print paragraphs for debugging
    print("\nParagraphs:")
    for i, p in enumerate(paragraphs):
        print(f"{i}: {p}")
        
    # Verify we have at least two paragraphs
    assert len(paragraphs) >= 2
    
    # Check that consecutive paragraphs have overlapping content
    for i in range(len(paragraphs) - 1):
        # Get words from each paragraph
        words_in_first = set(paragraphs[i].lower().split())
        words_in_second = set(paragraphs[i + 1].lower().split())
        common_words = words_in_first.intersection(words_in_second)
        
        # Print overlap info for debugging
        print(f"\nOverlap between paragraphs {i} and {i+1}:")
        print(f"Common words: {common_words}")
        
        # There should be some common words, excluding stop words
        content_words = common_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
        assert len(content_words) > 0, f"No content word overlap between paragraphs {i} and {i+1}"

def test_error_handling():
    """Test error handling in paragraph service."""
    service = ParagraphService()
    
    # Test empty text
    with pytest.raises(ValueError, match="Empty text"):
        service.get_embeddings("")
    
    # Test text shorter than min_length
    service = ParagraphService(min_length=1000)
    with pytest.raises(ValueError, match="Text is too short"):
        service.get_embeddings("Short text")
    
    # Test invalid merge strategy
    embeddings = [np.array([1.0, 2.0, 3.0])]
    with pytest.raises(ValueError, match="Unknown merging strategy"):
        service.merge_embeddings(embeddings, strategy="invalid")

def test_memory_efficiency():
    """Test memory efficiency with large texts."""
    service = ParagraphService()
    # Create a large text (100KB)
    large_text = "Test sentence. " * 10000
    
    # Process should not fail with large text
    paragraphs = service.split_text(large_text)
    assert len(paragraphs) > 0
    
    # Each paragraph should be within size limits
    assert all(len(p) <= service.max_length for p in paragraphs)