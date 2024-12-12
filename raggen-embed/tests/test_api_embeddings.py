import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock

from main import app
from api.models import ParagraphOptions

client = TestClient(app=app)

def test_embed_text_with_paragraphs():
    """Test embedding generation with paragraph processing."""
    text = (
        "First paragraph with some meaningful content. "
        "Second sentence here. "
        "Third sentence completes the paragraph. "
        "Second paragraph starts here with new content. "
        "More sentences to make it longer. "
        "Final sentence of the second paragraph."
    )
    
    # Mock embedding service and vector store
    with patch('api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embedding.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.dimension = 384
        mock_vs.add_vectors.return_value = [1]
        
        # Test request with paragraph options
        response = client.post(
            "/embed",
            json={
                "text": text,
                "paragraph_options": {
                    "enabled": True,
                    "max_length": 200,
                    "min_length": 50,
                    "overlap": 20,
                    "merge_strategy": "weighted"
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 384
        assert data["vector_id"] == 1
        assert data["text"] == text

def test_batch_embed_with_paragraphs():
    """Test batch embedding with paragraph processing."""
    texts = [
        "First text with multiple sentences. Second sentence here. Third sentence.",
        "Second text also has structure. More content here. Final part."
    ]
    
    # Mock embedding service and vector store
    with patch('api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embedding.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.dimension = 384
        mock_vs.add_vectors.return_value = [1, 2]
        
        # Test request with paragraph options
        response = client.post(
            "/embed/batch",
            json={
                "texts": texts,
                "paragraph_options": {
                    "enabled": True,
                    "max_length": 200,
                    "min_length": 50,
                    "overlap": 20,
                    "merge_strategy": "mean"
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == len(texts)
        for i, embedding_data in enumerate(data["embeddings"]):
            assert len(embedding_data["embedding"]) == 384
            assert embedding_data["vector_id"] == i + 1
            assert embedding_data["text"] == texts[i]

def test_invalid_paragraph_options():
    """Test validation of paragraph options."""
    text = "Sample text for testing."
    
    # Test invalid max_length
    response = client.post(
        "/embed",
        json={
            "text": text,
            "paragraph_options": {
                "enabled": True,
                "max_length": 50,  # Less than min_length
                "min_length": 100,
                "overlap": 20
            }
        }
    )
    assert response.status_code == 400
    
    # Test invalid overlap
    response = client.post(
        "/embed",
        json={
            "text": text,
            "paragraph_options": {
                "enabled": True,
                "max_length": 200,
                "min_length": 50,
                "overlap": 600  # Greater than max allowed
            }
        }
    )
    assert response.status_code == 400
    
    # Test invalid merge strategy
    response = client.post(
        "/embed",
        json={
            "text": text,
            "paragraph_options": {
                "enabled": True,
                "max_length": 200,
                "min_length": 50,
                "overlap": 20,
                "merge_strategy": "invalid"  # Invalid strategy
            }
        }
    )
    assert response.status_code == 400

def test_paragraph_processing_disabled():
    """Test that paragraph processing is properly disabled when not requested."""
    text = "Sample text for testing without paragraph processing."
    
    # Mock embedding service and vector store
    with patch('api.embeddings.EmbeddingService') as mock_embed_service, \
         patch('api.embeddings.FAISSVectorStore') as mock_store:
        
        # Setup mock embedding service
        mock_embed = mock_embed_service.return_value
        mock_embed.get_embedding.return_value = np.ones(384)
        
        # Setup mock vector store
        mock_vs = mock_store.return_value
        mock_vs.dimension = 384
        mock_vs.add_vectors.return_value = [1]
        
        # Test request without paragraph options
        response = client.post(
            "/embed",
            json={"text": text}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        
        # Verify that get_embedding was called only once
        mock_embed.get_embedding.assert_called_once_with(text)