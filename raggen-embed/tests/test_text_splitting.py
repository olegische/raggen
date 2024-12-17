"""Tests for text splitting functionality."""
import pytest
import numpy as np
from unittest.mock import Mock

from src.core.text_splitting import (
    TextSplitterService,
    TextSplitStrategy,
    SlidingWindowStrategy,
    ParagraphStrategy
)
from src.config.settings import TextSplitStrategy as StrategyType
from tests.conftest import SAMPLE_TEXT, SAMPLE_PARAGRAPHS


def test_service_creation_with_injected_strategy(
    mock_embedding_service,
    sliding_window_strategy,
    test_settings
):
    """Test creating service with injected strategy."""
    service = TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=sliding_window_strategy,
        settings=test_settings
    )
    
    assert service.embedding_service == mock_embedding_service
    assert service.split_strategy == sliding_window_strategy
    assert service.settings == test_settings


def test_service_creation_with_different_strategy(
    mock_embedding_service,
    paragraph_strategy,
    test_settings
):
    """Test creating service with different injected strategy."""
    service = TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=paragraph_strategy,
        settings=test_settings
    )
    
    assert isinstance(service.split_strategy, ParagraphStrategy)


def test_service_creation_without_embedding_service(
    sliding_window_strategy,
    test_settings
):
    """Test that service creation fails without embedding service."""
    with pytest.raises(ValueError, match="Embedding service must be provided"):
        TextSplitterService(
            embedding_service=None,
            split_strategy=sliding_window_strategy,
            settings=test_settings
        )


def test_split_text_with_sliding_window(text_splitter_service):
    """Test text splitting with sliding window strategy."""
    text = SAMPLE_TEXT
    chunks = text_splitter_service.split_text(text)
    
    assert len(chunks) > 0
    assert all(
        text_splitter_service.settings.text_min_length <= len(chunk) <= text_splitter_service.settings.text_max_length
        for chunk in chunks
    )


def test_split_text_with_paragraph_strategy(
    mock_embedding_service,
    paragraph_strategy,
    test_settings
):
    """Test text splitting with paragraph strategy."""
    service = TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=paragraph_strategy,
        settings=test_settings
    )
    
    chunks = service.split_text(SAMPLE_TEXT)
    assert len(chunks) == len(SAMPLE_PARAGRAPHS)
    assert all(chunk in SAMPLE_PARAGRAPHS for chunk in chunks)


def test_get_embeddings(text_splitter_service):
    """Test getting embeddings for text chunks."""
    embeddings = text_splitter_service.get_embeddings(SAMPLE_TEXT)
    
    assert isinstance(embeddings, np.ndarray)
    # Проверяем, что размерность векторов соответствует моку (3)
    assert embeddings.shape[1] == 3
    # Проверяем, что количество эмбеддингов соответствует количеству чанков
    assert embeddings.shape[0] == len(text_splitter_service.split_text(SAMPLE_TEXT))


def test_merge_embeddings_mean(text_splitter_service):
    """Test merging embeddings using mean strategy."""
    test_embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    merged = text_splitter_service.merge_embeddings(test_embeddings)
    expected = np.array([2.5, 3.5, 4.5])  # Mean of the vectors
    np.testing.assert_array_almost_equal(merged, expected)


def test_merge_embeddings_weighted(mock_embedding_service, sliding_window_strategy, test_settings):
    """Test merging embeddings using weighted strategy."""
    # Создаем настройки с weighted стратегией
    test_settings.embedding_merge_strategy = "weighted"
    
    service = TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=sliding_window_strategy,
        settings=test_settings
    )
    
    test_embeddings = np.array([
        [1.0, 1.0, 1.0],  # First vector (60% weight)
        [2.0, 2.0, 2.0]   # Second vector (40% weight)
    ])
    
    merged = service.merge_embeddings(test_embeddings)
    # First vector has 60% weight, second has 40%
    expected = np.array([1.4, 1.4, 1.4])
    np.testing.assert_array_almost_equal(merged, expected)


def test_merge_embeddings_invalid_strategy(
    mock_embedding_service,
    sliding_window_strategy,
    test_settings
):
    """Test that merging embeddings fails with invalid strategy."""
    # Создаем настройки с невалидной стратегией
    test_settings.embedding_merge_strategy = "invalid"
    
    service = TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=sliding_window_strategy,
        settings=test_settings
    )
    
    test_embeddings = np.array([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValueError, match="Unknown merging strategy: invalid"):
        service.merge_embeddings(test_embeddings)


def test_empty_text(text_splitter_service):
    """Test that empty text raises ValueError."""
    with pytest.raises(ValueError, match="Empty text"):
        text_splitter_service.split_text("")
    
    with pytest.raises(ValueError, match="Empty text"):
        text_splitter_service.get_embeddings("")


def test_empty_embeddings(text_splitter_service):
    """Test that empty embeddings array raises ValueError."""
    with pytest.raises(ValueError, match="No embeddings provided"):
        text_splitter_service.merge_embeddings(np.array([]))