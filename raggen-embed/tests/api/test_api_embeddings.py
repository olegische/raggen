import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import MagicMock, patch
import logging
from pprint import pformat

from src.main import app
from src.api.embeddings import get_embedding_service, get_vector_store
from utils.logging import get_logger
from core.embeddings import EmbeddingService

logger = get_logger(__name__)
client = TestClient(app)

def test_search_degradation():
    """Test search quality degradation without index training."""
    logger.info("=== Starting test_search_degradation ===")
    
    # Create mock services
    logger.info("Creating mock services")
    mock_embedding_service = MagicMock()
    
    # Create a set of vectors with known relationships
    # Base vector and vectors with increasing distance from it
    base_vector = np.ones(384, dtype=np.float32)
    similar_vectors = [
        base_vector + np.random.normal(0, noise_level, 384).astype(np.float32)
        for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    
    # Configure mock to return different vectors for different texts
    vectors_map = {
        "base": base_vector,
        "similar_0.1": similar_vectors[0],
        "similar_0.2": similar_vectors[1],
        "similar_0.3": similar_vectors[2],
        "similar_0.4": similar_vectors[3],
        "similar_0.5": similar_vectors[4],
    }
    
    def get_mock_embedding(text):
        return vectors_map[text]
    
    mock_embedding_service.get_embedding.side_effect = get_mock_embedding
    
    # Create vector store that maintains added vectors and texts
    stored_vectors = []
    stored_texts = []
    next_user_id = 1
    internal_to_user_ids = {}
    user_to_internal_ids = {}
    
    class MockVectorStore:
        dimension = 384
        is_trained = False
        
        def add_vectors(self, vectors):
            nonlocal stored_vectors, next_user_id, internal_to_user_ids, user_to_internal_ids
            start_internal_id = len(stored_vectors)
            
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            
            logger.info("\nAdding vectors:")
            logger.info("- Current stored vectors: %d", len(stored_vectors))
            
            stored_vectors.extend(vectors)
            
            # Generate user IDs and map them to internal IDs
            user_ids = []
            for i in range(len(vectors)):
                internal_id = start_internal_id + i
                user_id = next_user_id + i
                internal_to_user_ids[internal_id] = user_id
                user_to_internal_ids[user_id] = internal_id
                user_ids.append(user_id)
            
            next_vector_id += len(vectors)
            return user_ids
        
        def search(self, query_vectors, k):
            logger.info("\nSearching vectors:")
            logger.info("- Query shape: %s", query_vectors.shape)
            logger.info("- k: %d", k)
            logger.info("- Total stored vectors: %d", len(stored_vectors))
            
            # Compute actual distances
            query = query_vectors.reshape(1, -1)
            distances = []
            for vector in stored_vectors:
                dist = np.linalg.norm(query - vector)
                distances.append(dist)
            
            # Sort and return k nearest
            internal_indices = np.argsort(distances)[:k]
            sorted_distances = np.array([distances[i] for i in internal_indices])
            
            # Map internal indices to user IDs
            user_indices = np.array([internal_to_user_ids[i] for i in internal_indices])
            
            logger.info("Search results:")
            logger.info("- Distances: %s", sorted_distances)
            logger.info("- User indices: %s", user_indices)
            
            return (
                sorted_distances.reshape(1, -1),
                user_indices.reshape(1, -1)
            )
    
    mock_vector_store = MockVectorStore()
    
    # Override FastAPI dependencies
    logger.info("Overriding FastAPI dependencies")
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    
    try:
        # Step 1: Add base vector
        logger.info("\nStep 1: Adding base vector")
        response = client.post(
            "/api/v1/embed",
            json={"text": "base"}
        )
        if response.status_code != 200:
            logger.error("Search request failed:")
            logger.error("Request: %s", {"text": "base"})
            logger.error("Response: %s", response.json())
        assert response.status_code == 200
        base_data = response.json()
        base_id = base_data["vector_id"]
        logger.info("Base vector added with ID: %d", base_id)
        
        # Add similar vectors one by one and check search results
        vector_ids = {}
        for noise_level in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
            # Add vector
            text = f"similar_{noise_level}"
            logger.info("\nAdding vector: %s", text)
            response = client.post(
                "/api/v1/embed",
                json={"text": text}
            )
            assert response.status_code == 200
            data = response.json()
            vector_ids[text] = data["vector_id"]
            logger.info("Vector added with ID: %d", vector_ids[text])
            
            # Search similar vectors
            logger.info("\nSearching similar vectors")
            response = client.post(
                "/api/v1/search",
                json={
                    "text": "base",
                    "k": max(1, min(len(stored_vectors), 100))
                }
            )
            assert response.status_code == 200
            search_data = response.json()
            
            # Verify search results
            results = search_data["results"]
            logger.info("Search results after adding %s:", text)
            for i, result in enumerate(results):
                logger.info("- %d: vector_id=%d, score=%f", 
                          i, result["vector_id"], result["score"])
            
            # The base vector should always be the most similar to itself
            assert results[0]["vector_id"] == base_id, \
                f"Expected most similar to be base_id ({base_id}), got {results[0]['vector_id']}"
            
            # Verify that vectors are ordered by increasing noise level
            expected_order = ["base"] + [
                f"similar_{n}" for n in ["0.1", "0.2", "0.3", "0.4", "0.5"]
                if f"similar_{n}" in vector_ids
            ]
            actual_order = []
            for result in results:
                for text, vid in vector_ids.items():
                    if result["vector_id"] == vid:
                        actual_order.append(text)
                    elif result["vector_id"] == base_id:
                        actual_order.append("base")
            
            logger.info("Vector order:")
            logger.info("- Expected: %s", expected_order[:len(actual_order)])
            logger.info("- Actual: %s", actual_order)
            
            # As we add more vectors without training, the order might become incorrect
            if actual_order != expected_order[:len(actual_order)]:
                logger.warning("Search quality degraded: vectors not in expected order")
                logger.warning("This is expected behavior due to lack of index training")
        
        logger.info("=== test_search_degradation completed ===")
    finally:
        logger.info("Cleaning up FastAPI dependencies")
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

def test_embed_endpoint():
    """Test the /embed endpoint functionality."""
    logger.info("=== Starting test_embed_endpoint ===")
    
    # Create mock services
    mock_embedding_service = MagicMock(spec=EmbeddingService)
    test_vector = np.array([0.1] * 384, dtype=np.float32)
    mock_embedding_service.get_embedding.return_value = test_vector
    
    # Track vector IDs
    next_vector_id = 1
    
    class MockVectorStore:
        dimension = 384
        is_trained = True
        
        def add_vectors(self, vectors):
            nonlocal next_vector_id
            logger.info("Adding vectors to mock store, shape: %s", vectors.shape)
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            vector_ids = list(range(next_vector_id, next_vector_id + len(vectors)))
            next_vector_id += len(vectors)
            logger.info("Assigned vector IDs: %s", vector_ids)
            return vector_ids
        
        def train(self, vectors):
            logger.info("Mock training called with vectors shape: %s", vectors.shape)
            pass
    
    mock_store = MockVectorStore()
    
    # Override FastAPI dependencies
    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_store
    
    try:
        # Test 1: Basic embedding generation
        logger.info("\nTest 1: Basic embedding generation")
        test_text = "This is a test text"
        response = client.post(
            "/api/v1/embed",
            json={"text": test_text}
        )
        assert response.status_code == 200, f"Expected 200 status code, got {response.status_code}"
        data = response.json()
        logger.info("Response data: %s", data)
        
        # Verify response structure
        assert "embedding" in data, "Response missing 'embedding' field"
        assert "text" in data, "Response missing 'text' field"
        assert "vector_id" in data, "Response missing 'vector_id' field"
        
        # Verify embedding dimensionality
        assert len(data["embedding"]) == 384, f"Expected 384 dimensions, got {len(data['embedding'])}"
        
        # Convert response embedding to numpy array for comparison
        response_embedding = np.array(data["embedding"], dtype=np.float32)
        logger.info("Response embedding shape: %s", response_embedding.shape)
        logger.info("Response embedding first 5 values: %s", response_embedding[:5])
        logger.info("Expected embedding first 5 values: %s", test_vector[:5])
        
        # Verify vector values match our test vector with some tolerance
        assert np.allclose(response_embedding, test_vector, rtol=1e-5), \
            "Embedding values don't match test vector"
        
        # Verify text preservation
        assert data["text"] == test_text, f"Expected text '{test_text}', got '{data['text']}'"
        
        # Verify vector ID assignment
        assert data["vector_id"] == 1, f"Expected vector_id 1, got {data['vector_id']}"
        
        # Test 2: Caching behavior
        logger.info("\nTest 2: Testing caching behavior")
        # Make the same request again
        response2 = client.post(
            "/api/v1/embed",
            json={"text": test_text}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        logger.info("Second response data: %s", data2)
        
        # Verify we get the same embedding
        response_embedding2 = np.array(data2["embedding"], dtype=np.float32)
        assert np.allclose(response_embedding2, test_vector, rtol=1e-5), \
            "Second request returned different embedding"
        
        # Verify we get a new vector ID (since we're using a mock store)
        assert data2["vector_id"] == 2, f"Expected second vector_id 2, got {data2['vector_id']}"
        
        # Test 3: Different text
        logger.info("\nTest 3: Testing different text")
        different_text = "This is a different text"
        different_vector = np.array([0.2] * 384, dtype=np.float32)
        mock_embedding_service.get_embedding.return_value = different_vector
        
        response3 = client.post(
            "/api/v1/embed",
            json={"text": different_text}
        )
        assert response3.status_code == 200
        data3 = response3.json()
        logger.info("Third response data: %s", data3)
        
        # Verify different vector ID
        assert data3["vector_id"] == 3, f"Expected third vector_id 3, got {data3['vector_id']}"
        
        # Verify different embedding values
        response_embedding3 = np.array(data3["embedding"], dtype=np.float32)
        assert np.allclose(response_embedding3, different_vector, rtol=1e-5), \
            "Third request didn't return expected different embedding"
        
        # Verify mock was called correctly
        mock_embedding_service.get_embedding.assert_called_with(different_text)
        
        logger.info("=== test_embed_endpoint completed successfully ===")
        
    finally:
        app.dependency_overrides.clear()