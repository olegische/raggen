import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import MagicMock
import logging

from src.main import app
from src.api.documents import (
    get_paragraph_service,
    get_vector_store,
    ParagraphEmbeddingStrategy,
    MergedEmbeddingStrategy,
    CombinedEmbeddingStrategy
)

client = TestClient(app)
logger = logging.getLogger(__name__)

# Test data
SAMPLE_TEXT = (
    "First paragraph with some content.\n\n"
    "Second paragraph with different content.\n\n"
    "Third paragraph with more content."
)
SAMPLE_PARAGRAPHS = SAMPLE_TEXT.split("\n\n")
SAMPLE_EMBEDDING = np.ones((384,), dtype=np.float32)
SAMPLE_EMBEDDINGS = np.stack([SAMPLE_EMBEDDING for _ in range(len(SAMPLE_PARAGRAPHS))])

def test_paragraph_embedding_strategy():
    """Test ParagraphEmbeddingStrategy."""
    logger.info("Testing ParagraphEmbeddingStrategy")
    
    # Create mock services
    logger.info("Setting up mock services")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    mock_vs = MagicMock()
    mock_vs.add_vectors.return_value = [0, 1, 2]
    
    # Create strategy
    logger.info("Creating ParagraphEmbeddingStrategy instance")
    strategy = ParagraphEmbeddingStrategy(mock_ps, mock_vs)
    
    # Test process
    logger.info("Testing process method")
    result = strategy.process(SAMPLE_TEXT)
    
    # Verify results
    logger.info("Verifying results")
    assert result["strategy"] == "paragraphs"
    assert result["paragraphs_count"] == len(SAMPLE_PARAGRAPHS)
    assert result["vector_ids"] == [0, 1, 2]
    assert result["paragraphs"] == SAMPLE_PARAGRAPHS
    logger.info("ParagraphEmbeddingStrategy test completed successfully")

def test_merged_embedding_strategy():
    """Test MergedEmbeddingStrategy."""
    logger.info("Testing MergedEmbeddingStrategy")
    
    # Create mock services
    logger.info("Setting up mock services")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    merged_embedding = np.mean(SAMPLE_EMBEDDINGS, axis=0)
    mock_ps.merge_embeddings.return_value = merged_embedding
    
    mock_vs = MagicMock()
    mock_vs.add_vectors.return_value = [1]
    
    # Create strategy
    logger.info("Creating MergedEmbeddingStrategy instance")
    strategy = MergedEmbeddingStrategy(mock_ps, mock_vs)
    
    # Test process
    logger.info("Testing process method")
    result = strategy.process(SAMPLE_TEXT)
    
    # Verify results
    logger.info("Verifying results")
    assert result["strategy"] == "merged"
    assert result["paragraphs_count"] == len(SAMPLE_PARAGRAPHS)
    assert result["vector_id"] == 1
    assert result["paragraphs"] == SAMPLE_PARAGRAPHS
    logger.info("MergedEmbeddingStrategy test completed successfully")

def test_combined_embedding_strategy():
    """Test CombinedEmbeddingStrategy."""
    logger.info("Testing CombinedEmbeddingStrategy")
    
    # Create mock services
    logger.info("Setting up mock services")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    merged_embedding = np.mean(SAMPLE_EMBEDDINGS, axis=0)
    mock_ps.merge_embeddings.return_value = merged_embedding
    
    mock_vs = MagicMock()
    mock_vs.add_vectors.side_effect = [[1, 2, 3], [4]]
    
    # Create strategy
    logger.info("Creating CombinedEmbeddingStrategy instance")
    strategy = CombinedEmbeddingStrategy(mock_ps, mock_vs)
    
    # Test process
    logger.info("Testing process method")
    result = strategy.process(SAMPLE_TEXT)
    
    # Verify results
    logger.info("Verifying results")
    assert result["strategy"] == "combined"
    assert result["paragraphs_count"] == len(SAMPLE_PARAGRAPHS)
    assert result["paragraph_vector_ids"] == [1, 2, 3]
    assert result["merged_vector_id"] == 4
    assert result["paragraphs"] == SAMPLE_PARAGRAPHS
    logger.info("CombinedEmbeddingStrategy test completed successfully")