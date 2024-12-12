import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock, call
import logging

from src.main import app
from src.api.documents import (
    get_paragraph_service, 
    get_vector_store,
    TextProcessor,
    EmbeddingMerger,
    VectorStorer,
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
    mock_vs = MagicMock()
    mock_vs.add_vectors.side_effect = [[0, 1, 2], [3]]
    
    # Create storer
    logger.info("Creating VectorStorer instance")
    storer = VectorStorer()
    
    # Test store_vectors
    logger.info("Testing store_vectors method")
    vector_ids = storer.store_vectors(mock_vs, SAMPLE_EMBEDDINGS)
    assert vector_ids == [1, 2, 3]
    mock_vs.add_vectors.assert_called_with(SAMPLE_EMBEDDINGS)
    
    # Test store_single_vector
    logger.info("Testing store_single_vector method")
    vector_id = storer.store_single_vector(mock_vs, SAMPLE_EMBEDDING)
    assert vector_id == 4
    mock_vs.add_vectors.assert_called_with([SAMPLE_EMBEDDING])
    logger.info("VectorStorer test completed successfully")

def test_paragraph_embedding_strategy():
    """Test ParagraphEmbeddingStrategy."""
    logger.info("Testing ParagraphEmbeddingStrategy")
    
    # Create mock services
    logger.info("Setting up mock services")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    mock_vs = MagicMock()
    mock_vs.add_vectors.return_value = [1, 2, 3]
    
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
    assert result["vector_ids"] == [1, 2, 3]
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

def test_upload_document_paragraphs_strategy():
    """Test document upload with paragraphs strategy."""
    logger.info("Testing document upload with paragraphs strategy")
    
    # Create file content
    logger.info("Creating test file content")
    file_content = SAMPLE_TEXT.encode('utf-8')
    file = io.BytesIO(file_content)
    
    # Create mock services
    logger.info("Setting up mock services")
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    mock_vs = MagicMock()
    mock_vs.dimension = 384
    mock_vs.is_trained = True
    mock_vs.add_vectors = MagicMock(return_value=[1, 2, 3])  # Fixed: using MagicMock with list
    
    try:
        # Override FastAPI dependencies
        logger.info("Overriding FastAPI dependencies")
        app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        # Test request
        logger.info("Sending test request")
        response = client.post(
            "/api/v1/documents/upload",  # Fixed: added comma
            files={"file": ("test.txt", file, "text/plain")},  # Fixed: added commas
            params={"strategy": "paragraphs"}
        )
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Response error: {response.json()}")
            logger.error(f"Mock split_text calls: {mock_ps.split_text.mock_calls}")
            logger.error(f"Mock get_embeddings calls: {mock_ps.get_embeddings.mock_calls}")
            logger.error(f"Mock add_vectors calls: {mock_vs.add_vectors.mock_calls}")
        
        # Verify response
        logger.info("Verifying response")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document processed successfully"
        assert data["filename"] == "test.txt"
        assert data["strategy"] == "paragraphs"
        assert data["paragraphs_count"] == 3
        assert len(data["paragraphs"]) == 3
        assert data["vector_ids"] == [1, 2, 3]
        logger.info("Document upload test completed successfully")
        
    finally:
        logger.info("Cleaning up FastAPI dependencies")
        app.dependency_overrides.clear()

def test_upload_invalid_file_type():
    """Test uploading file with unsupported extension."""
    logger.info("Testing upload with invalid file type")
    content = "Some content"
    file = io.BytesIO(content.encode())
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", file, "application/pdf")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    logger.info("Invalid file type test completed successfully")

def test_upload_empty_file():
    """Test uploading empty file."""
    logger.info("Testing upload with empty file")
    file = io.BytesIO(b"")
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Error processing text" in response.json()["detail"]
    logger.info("Empty file test completed successfully")

def test_get_supported_types():
    """Test getting supported file types."""
    logger.info("Testing get supported types endpoint")
    response = client.get("/api/v1/documents/supported-types")
    assert response.status_code == 200
    data = response.json()
    assert "supported_types" in data
    assert "max_file_size_mb" in data
    assert "processing_strategies" in data
    assert isinstance(data["supported_types"], list)
    assert isinstance(data["max_file_size_mb"], float)
    assert isinstance(data["processing_strategies"], list)
    logger.info("Supported types test completed successfully")