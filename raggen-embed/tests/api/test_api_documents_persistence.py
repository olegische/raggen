import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import MagicMock, patch, call
import logging
from datetime import datetime

from src.main import create_app
from src.api.documents import (
    get_paragraph_service,
    get_vector_store,
    get_persistent_store,
    ProcessingStrategy
)
from src.core.paragraph_service import ParagraphService
from src.core.vector_store.persistent_store import PersistentStore
from src.core.vector_store.faiss_store import FAISSVectorStore
from src.core.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreType
from src.config.settings import Settings, get_settings
from conftest import (
    SAMPLE_TEXT,
    SAMPLE_PARAGRAPHS,
    SAMPLE_EMBEDDINGS,
)

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def mock_vector_store(test_settings):
    """Create mock vector store."""
    logger.info("[Test] Creating mock vector store")
    store = MagicMock(spec=FAISSVectorStore)
    store.settings = test_settings
    store.add = MagicMock()
    store.search = MagicMock(return_value=(np.array([]), np.array([])))
    store.save = MagicMock()
    store.load = MagicMock()
    
    def mock_add(vectors):
        logger.info("[Mock] Vector store add called with vectors shape: %s", vectors.shape)
        return None
    store.add.side_effect = mock_add
    
    return store

@pytest.fixture(scope="function")
def mock_persistent_store(test_settings, mock_vector_store):
    """Create mock persistent store."""
    logger.info("[Test] Creating mock persistent store")
    store = MagicMock(spec=PersistentStore)
    store.settings = test_settings
    store.index_path = test_settings.faiss_index_path
    store.store = mock_vector_store
    store.add = MagicMock()
    store.search = MagicMock(return_value=(np.array([]), np.array([])))
    
    def mock_add(vectors):
        logger.info("[Mock] Persistent store add called with vectors shape: %s", vectors.shape)
        store.store.add(vectors)  # Делегируем вызов базовому store
        return None
    store.add.side_effect = mock_add
    
    return store

@pytest.fixture(scope="function", autouse=True)
def mock_factory(mock_vector_store, mock_persistent_store):
    """Create mock factory."""
    logger.info("[Test] Creating mock factory")
    
    def mock_create(store_type, settings, force_new=False):
        logger.info("[Mock] Factory create called with store_type: %s", store_type)
        if store_type.value == VectorStoreType.FAISS.value:
            logger.info("[Mock] Returning mock vector store")
            return mock_vector_store
        elif store_type.value == VectorStoreType.PERSISTENT.value:
            logger.info("[Mock] Returning mock persistent store")
            return mock_persistent_store
        logger.error("[Mock] Unknown store type: %s", store_type)
        raise ValueError(f"Unknown store type: {store_type}")
    
    with patch('src.api.documents.VectorStoreFactory.create', side_effect=mock_create):
        yield

@pytest.fixture(scope="function")
def mock_paragraph_service():
    """Create mock paragraph service."""
    logger.info("[Test] Creating mock paragraph service")
    service = MagicMock()
    service.split_text = MagicMock(side_effect=lambda text: SAMPLE_PARAGRAPHS)
    service.get_embeddings = MagicMock(return_value=SAMPLE_EMBEDDINGS)
    service.merge_embeddings = MagicMock(side_effect=lambda embeddings: np.mean(embeddings, axis=0))
    
    return service

@pytest.fixture(scope="function")
def test_client(test_settings, mock_vector_store, mock_persistent_store, mock_paragraph_service):
    """Create test client with configured dependencies."""
    logger.info("[Test] Creating test client")
    
    # Create app
    app = create_app(settings=test_settings)
    
    # Configure dependency overrides before creating client
    logger.info("[Test] Setting up dependency overrides")
    
    # Set up dependency overrides
    app.dependency_overrides.update({
        get_settings: lambda: test_settings,
        get_paragraph_service: lambda: mock_paragraph_service,
        get_vector_store: lambda: mock_vector_store,
        get_persistent_store: lambda: mock_persistent_store
    })
    
    # Create and return client
    client = TestClient(app)
    logger.info("[Test] Test client created with overrides: %s", list(app.dependency_overrides.keys()))
    
    yield client
    
    # Clean up
    logger.info("[Test] Cleaning up dependency overrides")
    app.dependency_overrides.clear()

def test_store_creation(test_client, test_settings, mock_vector_store, mock_persistent_store):
    """Test store creation and settings propagation."""
    logger.info("[Test] Starting test_store_creation")
    
    # Upload a file to trigger store creation
    file = io.BytesIO(SAMPLE_TEXT.encode())
    logger.info("[Test] Sample text length: %d", len(SAMPLE_TEXT))
    
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    logger.info("[Test] Upload response status: %d", response.status_code)
    if response.status_code != 200:
        logger.error("[Test] Upload response content: %s", response.content)
    
    assert response.status_code == 200
    
    # Verify settings propagation
    assert mock_vector_store.settings == test_settings
    assert mock_persistent_store.settings == test_settings
    
    # Verify store interactions
    logger.info("[Test] Mock persistent store add call count: %d", mock_persistent_store.add.call_count)
    mock_persistent_store.add.assert_called()

def test_persistence(test_client, mock_persistent_store, test_settings):
    """Test vector store persistence between requests."""
    logger.info("[Test] Testing vector store persistence")
    
    # First request
    file1 = io.BytesIO(SAMPLE_TEXT.encode())
    response1 = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test1.txt", file1, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response1.status_code == 200
    
    # Second request
    file2 = io.BytesIO(SAMPLE_TEXT.encode())
    response2 = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test2.txt", file2, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response2.status_code == 200
    
    # Verify store was used
    assert mock_persistent_store.add.call_count == 2

def test_upload_invalid_file_type(test_client):
    """Test uploading file with unsupported extension."""
    logger.info("[Test] Testing upload with invalid file type")
    
    content = "Some content"
    file = io.BytesIO(content.encode())
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", file, "application/pdf")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    logger.info("[Test] Invalid file type test completed successfully")

def test_upload_empty_file(test_client):
    """Test uploading empty file."""
    logger.info("[Test] Testing upload with empty file")
    
    file = io.BytesIO(b"")
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "Empty text" in response.json()["detail"]
    logger.info("[Test] Empty file test completed successfully")

def test_get_supported_types(test_client):
    """Test getting supported file types."""
    logger.info("[Test] Testing get supported types endpoint")
    response = test_client.get("/api/v1/documents/supported-types")
    assert response.status_code == 200
    data = response.json()
    assert "supported_types" in data
    assert "max_file_size_mb" in data
    assert "processing_strategies" in data
    assert isinstance(data["supported_types"], list)
    assert isinstance(data["max_file_size_mb"], float)
    assert isinstance(data["processing_strategies"], list)
    logger.info("[Test] Supported types test completed successfully")

def test_persistence_with_different_strategies(test_client, mock_persistent_store, mock_paragraph_service):
    """Test persistence with different processing strategies."""
    logger.info("[Test] Testing persistence with different strategies")
    
    strategies = ["paragraphs", "merged", "combined"]
    for strategy in strategies:
        # Upload document with current strategy
        file = io.BytesIO(SAMPLE_TEXT.encode())
        response = test_client.post(
            "/api/v1/documents/upload",
            files={"file": (f"test_{strategy}.txt", file, "text/plain")},
            params={"strategy": strategy}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == strategy
        
        # Verify vectors are accessible after reload
        file2 = io.BytesIO(SAMPLE_TEXT.encode())
        response2 = test_client.post(
            "/api/v1/documents/upload",
            files={"file": (f"test_{strategy}_2.txt", file2, "text/plain")},
            params={"strategy": strategy}
        )
        assert response2.status_code == 200
    
    # Verify store was used appropriately for each strategy
    expected_add_calls = {
        "paragraphs": 2,  # Один вызов на файл, все параграфы добавляются вместе матрицей (3, 384)
        "merged": 2,  # Один вызов на файл с одним вектором (1, 384)
        "combined": 4  # Два вызова на файл: параграфы (3, 384) + объединенный (1, 384)
    }
    total_expected_calls = sum(expected_add_calls.values())
    actual_calls = mock_persistent_store.add.call_count
    logger.info("[Test] Expected %d calls, got %d calls", total_expected_calls, actual_calls)
    logger.info("[Test] Expected calls breakdown: %s", expected_add_calls)
    logger.info("[Test] Actual calls:")
    for i, call in enumerate(mock_persistent_store.add.call_args_list):
        args, kwargs = call
        vectors = args[0]
        logger.info("[Test] Call %d: shape %s", i + 1, vectors.shape)
    
    assert mock_persistent_store.add.call_count == total_expected_calls
    
    # Verify paragraph service was used correctly for each strategy
    expected_split_text_calls = {
        "paragraphs": 2,  # Два вызова для paragraphs (по одному на файл)
        "merged": 2,      # Два вызова для merged (по одному на файл)
        "combined": 2     # Два вызова для combined (по одному на файл)
    }
    total_split_text_calls = sum(expected_split_text_calls.values())
    assert mock_paragraph_service.split_text.call_count == total_split_text_calls
    assert mock_paragraph_service.get_embeddings.call_count == total_split_text_calls

def test_error_handling(test_client, mock_persistent_store):
    """Test error handling in document processing."""
    logger.info("[Test] Testing error handling")
    
    # Test file size limit
    large_content = "x" * (10 * 1024 * 1024 + 1)  # Slightly over 10MB
    file = io.BytesIO(large_content.encode())
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", file, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]
    
    # Test store error handling
    mock_persistent_store.add.side_effect = Exception("Store error")
    file = io.BytesIO(SAMPLE_TEXT.encode())
    response = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", file, "text/plain")},
        params={"strategy": "paragraphs"}
    )
    assert response.status_code == 500
    assert "Error processing document" in response.json()["detail"]