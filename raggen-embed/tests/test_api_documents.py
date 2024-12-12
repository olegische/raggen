import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

from src.main import app
from src.api.documents import get_paragraph_service, get_vector_store

client = TestClient(app)

def test_upload_document_paragraphs_strategy():
    """Test document upload with paragraphs strategy."""
    # Mock file content
    content = (
        "First paragraph with some content.\n\n"
        "Second paragraph with different content.\n\n"
        "Third paragraph with more content."
    )
    
    # Mock file
    file = MagicMock()
    file.filename = "test.txt"
    file.read.return_value = content.encode()
    
    # Mock services
    with patch('src.api.documents.ParagraphService') as mock_paragraph_service, \
         patch('src.api.documents.FAISSVectorStore') as mock_store:
        
        # Setup mock paragraph service
        mock_ps = mock_paragraph_service.return_value
        mock_ps.split_text.return_value = content.split("\n\n")
        mock_ps.get_embeddings.return_value = [np.ones(384) for _ in range(3)]
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.add_vectors.return_value = [1, 2, 3]
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", content.encode(), "text/plain")},
                params={"strategy": "paragraphs"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Document processed successfully"
            assert data["filename"] == "test.txt"
            assert data["strategy"] == "paragraphs"
            assert data["paragraphs_count"] == 3
            assert len(data["paragraphs"]) == 3
            assert data["vector_ids"] == [1, 2, 3]
            
            # Verify service calls
            mock_ps.split_text.assert_called_once()
            mock_ps.get_embeddings.assert_called_once()
            mock_vs.add_vectors.assert_called_once()
        finally:
            app.dependency_overrides.clear()

def test_upload_document_merged_strategy():
    """Test document upload with merged strategy."""
    content = (
        "First paragraph with some content.\n\n"
        "Second paragraph with different content."
    )
    
    file = MagicMock()
    file.filename = "test.txt"
    file.read.return_value = content.encode()
    
    with patch('src.api.documents.ParagraphService') as mock_paragraph_service, \
         patch('src.api.documents.FAISSVectorStore') as mock_store:
        
        # Setup mock paragraph service
        mock_ps = mock_paragraph_service.return_value
        paragraphs = content.split("\n\n")
        mock_ps.split_text.return_value = paragraphs
        mock_ps.get_embeddings.return_value = [np.ones(384) for _ in paragraphs]
        mock_ps.merge_embeddings.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.add_vectors.return_value = [1]
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", content.encode(), "text/plain")},
                params={"strategy": "merged"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Document processed successfully"
            assert data["filename"] == "test.txt"
            assert data["strategy"] == "merged"
            assert data["paragraphs_count"] == 2
            assert len(data["paragraphs"]) == 2
            assert data["vector_id"] == 1
            
            # Verify service calls
            mock_ps.split_text.assert_called_once()
            mock_ps.get_embeddings.assert_called_once()
            mock_ps.merge_embeddings.assert_called_once()
            mock_vs.add_vectors.assert_called_once()
        finally:
            app.dependency_overrides.clear()

def test_upload_document_combined_strategy():
    """Test document upload with combined strategy."""
    content = (
        "First paragraph with some content.\n\n"
        "Second paragraph with different content."
    )
    
    file = MagicMock()
    file.filename = "test.txt"
    file.read.return_value = content.encode()
    
    with patch('src.api.documents.ParagraphService') as mock_paragraph_service, \
         patch('src.api.documents.FAISSVectorStore') as mock_store:
        
        # Setup mock paragraph service
        mock_ps = mock_paragraph_service.return_value
        paragraphs = content.split("\n\n")
        mock_ps.split_text.return_value = paragraphs
        mock_ps.get_embeddings.return_value = [np.ones(384) for _ in paragraphs]
        mock_ps.merge_embeddings.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.add_vectors.side_effect = [[1, 2], [3]]  # First call for paragraphs, second for merged
        
        # Override FastAPI dependencies
        app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        
        try:
            # Test request
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", content.encode(), "text/plain")},
                params={"strategy": "combined"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Document processed successfully"
            assert data["filename"] == "test.txt"
            assert data["strategy"] == "combined"
            assert data["paragraphs_count"] == 2
            assert len(data["paragraphs"]) == 2
            assert data["paragraph_vector_ids"] == [1, 2]
            assert data["merged_vector_id"] == 3
            
            # Verify service calls
            mock_ps.split_text.assert_called_once()
            mock_ps.get_embeddings.assert_called_once()
            mock_ps.merge_embeddings.assert_called_once()
            assert mock_vs.add_vectors.call_count == 2
        finally:
            app.dependency_overrides.clear()

def test_upload_invalid_file_type():
    """Test uploading file with unsupported extension."""
    content = "Some content"
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", content.encode(), "application/pdf")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_empty_file():
    """Test uploading empty file."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", b"", "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Error processing text" in response.json()["detail"]

def test_get_supported_types():
    """Test getting supported file types."""
    response = client.get("/api/v1/documents/supported-types")
    assert response.status_code == 200
    data = response.json()
    assert "supported_types" in data
    assert "max_file_size_mb" in data
    assert "processing_strategies" in data
    assert isinstance(data["supported_types"], list)
    assert isinstance(data["max_file_size_mb"], float)
    assert isinstance(data["processing_strategies"], list)