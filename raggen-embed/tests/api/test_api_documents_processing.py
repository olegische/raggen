import numpy as np
from unittest.mock import MagicMock
import logging

from src.api.documents import (
    TextProcessor,
    EmbeddingMerger,
    VectorStorer
)
from src.core.vector_store.persistent_store import PersistentFAISSStore

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

def test_text_processor():
    """Test TextProcessor component."""
    logger.info("Testing TextProcessor component")
    
    # Create mock paragraph service
    logger.info("Setting up mock paragraph service")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    # Create text processor
    logger.info("Creating TextProcessor instance")
    processor = TextProcessor(mock_ps)
    
    # Test process_text
    logger.info("Testing process_text method")
    paragraphs, embeddings = processor.process_text(SAMPLE_TEXT)
    
    # Verify results
    logger.info("Verifying results")
    assert paragraphs == SAMPLE_PARAGRAPHS
    assert np.array_equal(embeddings, SAMPLE_EMBEDDINGS)
    mock_ps.split_text.assert_called_once_with(SAMPLE_TEXT)
    mock_ps.get_embeddings.assert_called_once_with(SAMPLE_TEXT)
    logger.info("TextProcessor test completed successfully")

def test_embedding_merger():
    """Test EmbeddingMerger component."""
    logger.info("Testing EmbeddingMerger component")
    
    # Create mock paragraph service
    logger.info("Setting up mock paragraph service")
    mock_ps = MagicMock()
    merged_embedding = np.mean(SAMPLE_EMBEDDINGS, axis=0)
    mock_ps.merge_embeddings.return_value = merged_embedding
    
    # Create merger
    logger.info("Creating EmbeddingMerger instance")
    merger = EmbeddingMerger()
    
    # Test merge_embeddings
    logger.info("Testing merge_embeddings method")
    result = merger.merge_embeddings(mock_ps, SAMPLE_EMBEDDINGS)
    
    # Verify results
    logger.info("Verifying results")
    assert np.array_equal(result, merged_embedding)
    mock_ps.merge_embeddings.assert_called_once_with(SAMPLE_EMBEDDINGS)
    logger.info("EmbeddingMerger test completed successfully")

def test_vector_storer():
    """Test VectorStorer component."""
    logger.info("Testing VectorStorer component")
    
    # Create mock vector store
    logger.info("Setting up mock vector store")
    mock_vs = MagicMock(spec=PersistentFAISSStore)
    mock_vs.add_vectors.side_effect = [[0, 1, 2], [3]]
    
    # Create storer
    logger.info("Creating VectorStorer instance")
    storer = VectorStorer()
    
    # Test store_vectors
    logger.info("Testing store_vectors method")
    vector_ids = storer.store_vectors(mock_vs, SAMPLE_EMBEDDINGS)
    assert vector_ids == [0, 1, 2]
    mock_vs.add_vectors.assert_called_with(SAMPLE_EMBEDDINGS)
    
    # Test store_single_vector
    logger.info("Testing store_single_vector method")
    vector_id = storer.store_single_vector(mock_vs, SAMPLE_EMBEDDING)
    assert vector_id == 3
    mock_vs.add_vectors.assert_called_with([SAMPLE_EMBEDDING])
    logger.info("VectorStorer test completed successfully")