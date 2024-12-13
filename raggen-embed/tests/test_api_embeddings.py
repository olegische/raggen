import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock
import logging
from pprint import pformat

from src.main import app
from src.api.embeddings import get_embedding_service, get_vector_store
from utils.logging import get_logger

logger = get_logger(__name__)
client = TestClient(app)

# Test data
SAMPLE_TEXT = "Sample text for embedding generation"
SAMPLE_TEXTS = [
    "First text for embedding",
    "Second text for embedding"
]
SAMPLE_EMBEDDING = np.ones(384, dtype=np.float32)
SAMPLE_EMBEDDINGS = np.ones((len(SAMPLE_TEXTS), 384), dtype=np.float32)

def test_embed_text():
    """Test embedding generation for a single text."""
    logger.info("=== Starting test_embed_text ===")
    
    # Create mock services
    logger.info("Creating mock services")
    mock_embedding_service = MagicMock()
    mock_embedding_service.get_embedding.return_value = SAMPLE_EMBEDDING
    
    mock_vector_store = MagicMock()
    mock_vector_store.dimension = 384
    mock_vector_store.is_trained = True
    mock_vector_store.add_vectors.return_value = [0]  # ID starts from 0
    
    logger.info("Mock configuration:")
    logger.info("- embedding shape: %s", SAMPLE_EMBEDDING.shape)
    logger.info("- vector_store.add_vectors return value: %s", mock_vector_store.add_vectors.return_value)
    
    # Override FastAPI dependencies
    logger.info("Overriding FastAPI dependencies")
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    
    try:
        # Test request
        logger.info("Sending test request")
        logger.info("Request data: %s", {"text": SAMPLE_TEXT})
        response = client.post(
            "/api/v1/embed",
            json={"text": SAMPLE_TEXT}
        )
        
        logger.info("Response received:")
        logger.info("- status code: %d", response.status_code)
        logger.info("- headers:\n%s", pformat(dict(response.headers)))
        logger.info("- body:\n%s", pformat(response.text))
        
        # Parse and validate response
        logger.info("Validating response")
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        
        data = response.json()
        logger.info("Parsed response data:\n%s", pformat(data))
        
        # Validate response structure
        logger.info("Checking response structure")
        assert "embedding" in data, "Response missing 'embedding' field"
        assert "vector_id" in data, "Response missing 'vector_id' field"
        assert "text" in data, "Response missing 'text' field"
        
        # Validate embedding
        logger.info("Checking embedding")
        assert len(data["embedding"]) == 384, f"Expected embedding length 384, got {len(data['embedding'])}"
        
        # Validate vector ID
        logger.info("Checking vector ID")
        logger.info("Expected vector_id: %d", mock_vector_store.add_vectors.return_value[0])
        logger.info("Actual vector_id: %d", data["vector_id"])
        assert data["vector_id"] == 0, f"Expected vector_id 0, got {data['vector_id']}"
        
        # Verify service calls
        logger.info("Verifying service calls")
        mock_embedding_service.get_embedding.assert_called_once_with(SAMPLE_TEXT)
        mock_vector_store.add_vectors.assert_called_once()
        
        logger.info("Mock service call history:")
        logger.info("- embedding_service.get_embedding calls:\n%s", 
                   pformat(mock_embedding_service.get_embedding.mock_calls))
        logger.info("- vector_store.add_vectors calls:\n%s", 
                   pformat(mock_vector_store.add_vectors.mock_calls))
        logger.info("- embedding_service.get_embedding return value:\n%s", 
                   pformat(mock_embedding_service.get_embedding.return_value))
        logger.info("- vector_store.add_vectors return value:\n%s", 
                   pformat(mock_vector_store.add_vectors.return_value))
        
        # Log actual call arguments
        logger.info("Mock service call arguments:")
        logger.info("- embedding_service.get_embedding args:\n%s", 
                   pformat(mock_embedding_service.get_embedding.call_args))
        logger.info("- vector_store.add_vectors args:\n%s", 
                   pformat(mock_vector_store.add_vectors.call_args))
        
        logger.info("=== test_embed_text completed successfully ===")
    finally:
        logger.info("Cleaning up FastAPI dependencies")
        app.dependency_overrides.clear()

def test_batch_embed():
    """Test batch embedding generation."""
    logger.info("Starting test_batch_embed")
    
    # Create mock services
    logger.info("Creating mock services")
    mock_embedding_service = MagicMock()
    mock_embedding_service.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    mock_vector_store = MagicMock()
    mock_vector_store.dimension = 384
    mock_vector_store.is_trained = True
    mock_vector_store.add_vectors.return_value = [0, 1]  # IDs start from 0
    
    logger.info("Mock configuration:")
    logger.info("- embeddings shape: %s", SAMPLE_EMBEDDINGS.shape)
    logger.info("- vector_store.add_vectors return value: %s", mock_vector_store.add_vectors.return_value)
    
    # Override FastAPI dependencies
    logger.info("Overriding FastAPI dependencies")
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    
    try:
        # Test request
        logger.info("Sending test request")
        logger.info("Request data: %s", {"texts": SAMPLE_TEXTS})
        response = client.post(
            "/api/v1/embed/batch",
            json={"texts": SAMPLE_TEXTS}
        )
        
        logger.info("Response received:")
        logger.info("- status code: %d", response.status_code)
        logger.info("- headers:\n%s", pformat(dict(response.headers)))
        logger.info("- body:\n%s", pformat(response.text))
        
        # Parse and validate response
        logger.info("Validating response")
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        
        data = response.json()
        logger.info("Parsed response data:\n%s", pformat(data))
        
        # Validate response structure
        logger.info("Checking response structure")
        assert "embeddings" in data, "Response missing 'embeddings' field"
        assert len(data["embeddings"]) == len(SAMPLE_TEXTS), \
            f"Expected {len(SAMPLE_TEXTS)} embeddings, got {len(data['embeddings'])}"
        
        # Validate each embedding
        for i, embedding_data in enumerate(data["embeddings"]):
            logger.info("Checking embedding %d", i)
            assert len(embedding_data["embedding"]) == 384, \
                f"Expected embedding length 384, got {len(embedding_data['embedding'])}"
            assert embedding_data["vector_id"] == i, \
                f"Expected vector_id {i}, got {embedding_data['vector_id']}"
            assert embedding_data["text"] == SAMPLE_TEXTS[i], \
                f"Expected text '{SAMPLE_TEXTS[i]}', got '{embedding_data['text']}'"
        
        # Verify service calls
        logger.info("Verifying service calls")
        mock_embedding_service.get_embeddings.assert_called_once_with(SAMPLE_TEXTS)
        mock_vector_store.add_vectors.assert_called_once_with(SAMPLE_EMBEDDINGS)
        
        logger.info("Mock service call history:")
        logger.info("- embedding_service.get_embeddings calls:\n%s", 
                   pformat(mock_embedding_service.get_embeddings.mock_calls))
        logger.info("- vector_store.add_vectors calls:\n%s", 
                   pformat(mock_vector_store.add_vectors.mock_calls))
        
        logger.info("test_batch_embed completed successfully")
    finally:
        app.dependency_overrides.clear()

def test_invalid_text():
    """Test validation of text input."""
    logger.info("Starting test_invalid_text")
    
    # Test empty text
    logger.info("Testing empty text")
    response = client.post(
        "/api/v1/embed",
        json={"text": ""}
    )
    logger.info("Empty text response: %s", response.json())
    assert response.status_code == 422  # FastAPI validation error
    
    # Test text too long
    logger.info("Testing text too long")
    response = client.post(
        "/api/v1/embed",
        json={"text": "x" * 513}  # More than 512 characters
    )
    logger.info("Long text response: %s", response.json())
    assert response.status_code == 422  # FastAPI validation error
    
    logger.info("test_invalid_text completed successfully")

def test_invalid_batch():
    """Test validation of batch input."""
    logger.info("Starting test_invalid_batch")
    
    # Test empty texts list
    logger.info("Testing empty texts list")
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": []}
    )
    logger.info("Empty list response: %s", response.json())
    assert response.status_code == 422  # FastAPI validation error
    
    # Test too many texts
    logger.info("Testing too many texts")
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": ["text"] * 33}  # More than 32 texts
    )
    logger.info("Too many texts response: %s", response.json())
    assert response.status_code == 422  # FastAPI validation error
    
    # Test text too long in batch
    logger.info("Testing text too long in batch")
    response = client.post(
        "/api/v1/embed/batch",
        json={"texts": ["x" * 513]}  # Text longer than 512 characters
    )
    logger.info("Long text in batch response: %s", response.json())
    assert response.status_code == 422  # FastAPI validation error
    
    logger.info("test_invalid_batch completed successfully")

def test_search_similar():
    """Test similarity search."""
    logger.info("Starting test_search_similar")
    query = "Sample query text"
    
    # Create mock services
    logger.info("Creating mock services")
    mock_embedding_service = MagicMock()
    mock_embedding_service.get_embedding.return_value = SAMPLE_EMBEDDING
    
    mock_vector_store = MagicMock()
    mock_vector_store.dimension = 384
    mock_vector_store.is_trained = True
    mock_vector_store.search.return_value = (
        np.array([[0.2, 0.3]]),  # distances
        np.array([[0, 1]])       # indices start from 0
    )
    
    logger.info("Mock configuration:")
    logger.info("- embedding shape: %s", SAMPLE_EMBEDDING.shape)
    logger.info("- search return values:")
    logger.info("  - distances: %s", mock_vector_store.search.return_value[0])
    logger.info("  - indices: %s", mock_vector_store.search.return_value[1])
    
    # Override FastAPI dependencies
    logger.info("Overriding FastAPI dependencies")
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    
    try:
        # Test request
        logger.info("Sending test request")
        logger.info("Request data: %s", {"text": query, "k": 2})
        response = client.post(
            "/api/v1/search",
            json={
                "text": query,
                "k": 2
            }
        )
        
        logger.info("Response received:")
        logger.info("- status code: %d", response.status_code)
        logger.info("- headers:\n%s", pformat(dict(response.headers)))
        logger.info("- body:\n%s", pformat(response.text))
        
        # Parse and validate response
        logger.info("Validating response")
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        
        data = response.json()
        logger.info("Parsed response data:\n%s", pformat(data))
        
        # Validate response structure
        logger.info("Checking response structure")
        assert data["query"] == query, f"Expected query '{query}', got '{data['query']}'"
        assert len(data["results"]) == 2, f"Expected 2 results, got {len(data['results'])}"
        
        # Verify scores are normalized correctly
        logger.info("Checking scores")
        assert data["results"][0]["score"] > data["results"][1]["score"], \
            "First result should have higher score"
        assert 0 <= data["results"][0]["score"] <= 1, \
            f"Score should be between 0 and 1, got {data['results'][0]['score']}"
        assert 0 <= data["results"][1]["score"] <= 1, \
            f"Score should be between 0 and 1, got {data['results'][1]['score']}"
        
        # Verify vector IDs match indices
        logger.info("Checking vector IDs")
        assert data["results"][0]["vector_id"] == 0, \
            f"Expected vector_id 0, got {data['results'][0]['vector_id']}"
        assert data["results"][1]["vector_id"] == 1, \
            f"Expected vector_id 1, got {data['results'][1]['vector_id']}"
        
        # Verify service calls
        logger.info("Verifying service calls")
        mock_embedding_service.get_embedding.assert_called_once_with(query)
        mock_vector_store.search.assert_called_once()
        
        logger.info("Mock service call history:")
        logger.info("- embedding_service.get_embedding calls:\n%s", 
                   pformat(mock_embedding_service.get_embedding.mock_calls))
        logger.info("- vector_store.search calls:\n%s", 
                   pformat(mock_vector_store.search.mock_calls))
        
        logger.info("test_search_similar completed successfully")
    finally:
        app.dependency_overrides.clear()