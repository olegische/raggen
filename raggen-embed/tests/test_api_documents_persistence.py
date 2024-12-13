import io
import os
import pytest
import tempfile
import shutil
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import MagicMock, patch
import logging
from datetime import datetime

from src.main import app
from src.api.documents import (
    get_paragraph_service,
    get_vector_store
)
from src.core.vector_store.persistent_store import PersistentFAISSStore
from src.core.vector_store.faiss_store import FAISSVectorStore

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

@pytest.fixture
def test_dir():
    """Create temporary directory for vector store files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mocked_vector_store():
    with patch("os.makedirs") as mock_makedirs, \
         patch("os.path.exists") as mock_exists, \
         patch("os.listdir") as mock_listdir, \
         patch("os.remove") as mock_remove, \
         patch("os.rename") as mock_rename, \
         patch("os.path.join", side_effect=os.path.join) as mock_join, \
         patch("src.core.vector_store.faiss_store.FAISSVectorStore.save") as mock_save, \
         patch("src.core.vector_store.faiss_store.FAISSVectorStore.load") as mock_load, \
         patch("src.core.vector_store.faiss_store.FAISSVectorStore.add") as mock_add:
        
        # Настраиваем мок для load, возвращающий фиктивный FAISSStore
        mock_store = MagicMock()
        mock_store.dimension = 384
        mock_store.add = mock_add
        mock_store.save = mock_save
        mock_load.return_value = mock_store
        
        # Настраиваем поведение файловой системы
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "index_20240101_120000.faiss",
            "index_20240101_120001.faiss",
            "index_20240101_120002.faiss"
        ]
        
        yield {
            "mock_makedirs": mock_makedirs,
            "mock_exists": mock_exists,
            "mock_listdir": mock_listdir,
            "mock_remove": mock_remove,
            "mock_rename": mock_rename,
            "mock_join": mock_join,
            "mock_load": mock_load,
            "mock_save": mock_save,
            "mock_add": mock_add,
            "mock_store": mock_store
        }

def test_add_vectors_with_mocks(mocked_vector_store):
    """Test add_vectors method with mocked dependencies."""
    # Настраиваем моки
    mock_exists = mocked_vector_store["mock_exists"]
    mock_save = mocked_vector_store["mock_save"]
    mock_add = mocked_vector_store["mock_add"]
    mock_load = mocked_vector_store["mock_load"]
    mock_rename = mocked_vector_store["mock_rename"]
    mock_listdir = mocked_vector_store["mock_listdir"]
    
    # Создаем тестовые данные
    test_vectors = np.random.rand(3, 384).astype(np.float32)
    index_path = "/fake/path/index.faiss"
    
    # Создаем экземпляр PersistentFAISSStore
    store = PersistentFAISSStore(index_path=index_path)
    
    # Добавляем векторы
    store.add_vectors(test_vectors)
    
    # Проверяем вызовы
    mock_add.assert_called_once_with(test_vectors)  # Проверяем, что векторы были добавлены
    mock_save.assert_called_once_with(index_path)   # Проверяем сохранение индекса
    mock_rename.assert_called_once()                # Проверяем создание бэкапа
    mock_listdir.assert_called_once()              # Проверяем чтение директории для очистки бэкапов

def test_persistence(mocked_vector_store):
    """Test vector store persistence between requests."""
    logger.info("Testing vector store persistence")
    
    # Настраиваем моки
    mock_load = mocked_vector_store["mock_load"]
    mock_save = mocked_vector_store["mock_save"]
    mock_add = mocked_vector_store["mock_add"]
    
    # Create mock paragraph service
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    
    # Настраиваем FastAPI зависимости
    app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
    app.dependency_overrides[get_vector_store] = lambda: PersistentFAISSStore(index_path="/fake/path/index.faiss")
    
    try:
        # Первый запрос
        file1 = io.BytesIO(SAMPLE_TEXT.encode())
        response1 = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test1.txt", file1, "text/plain")},
            params={"strategy": "paragraphs"}
        )
        assert response1.status_code == 200
        vector_ids1 = response1.json()["vector_ids"]
        
        # Симулируем перезапуск
        app.dependency_overrides[get_vector_store] = lambda: PersistentFAISSStore(index_path="/fake/path/index.faiss")
        
        # Второй запрос
        file2 = io.BytesIO(SAMPLE_TEXT.encode())
        response2 = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test2.txt", file2, "text/plain")},
            params={"strategy": "paragraphs"}
        )
        assert response2.status_code == 200
        vector_ids2 = response2.json()["vector_ids"]
        
        # Проверяем, что новые вектора добавлены
        assert min(vector_ids2) > max(vector_ids1)
        
        # Проверяем вызовы моков
        assert mock_add.call_count == 2  # Два вызова add для двух запросов
        assert mock_save.call_count == 2  # Два вызова save
        assert mock_load.call_count == 2  # Два вызова load
        
    finally:
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

def test_persistence_with_different_strategies(mocked_vector_store):
    """Test persistence with different processing strategies."""
    logger.info("Testing persistence with different strategies")
    
    # Настраиваем моки
    mock_load = mocked_vector_store["mock_load"]
    mock_save = mocked_vector_store["mock_save"]
    mock_add = mocked_vector_store["mock_add"]
    
    # Create mock paragraph service
    mock_ps = MagicMock()
    mock_ps.split_text.return_value = SAMPLE_PARAGRAPHS
    mock_ps.get_embeddings.return_value = SAMPLE_EMBEDDINGS
    mock_ps.merge_embeddings.return_value = np.mean(SAMPLE_EMBEDDINGS, axis=0)
    
    # Настраиваем FastAPI зависимости
    app.dependency_overrides[get_paragraph_service] = lambda: mock_ps
    app.dependency_overrides[get_vector_store] = lambda: PersistentFAISSStore(index_path="/fake/path/index.faiss")
    
    try:
        # Test each strategy
        strategies = ["paragraphs", "merged", "combined"]
        for strategy in strategies:
            # Upload document with current strategy
            file = io.BytesIO(SAMPLE_TEXT.encode())
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": (f"test_{strategy}.txt", file, "text/plain")},
                params={"strategy": strategy}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["strategy"] == strategy
            
            # Симулируем перезапуск для каждой стратегии
            app.dependency_overrides[get_vector_store] = lambda: PersistentFAISSStore(index_path="/fake/path/index.faiss")
            
            # Verify vectors are accessible after reload
            file2 = io.BytesIO(SAMPLE_TEXT.encode())
            response2 = client.post(
                "/api/v1/documents/upload",
                files={"file": (f"test_{strategy}_2.txt", file2, "text/plain")},
                params={"strategy": strategy}
            )
            assert response2.status_code == 200
        
        # Проверяем вызовы моков
        expected_load_calls = len(strategies) * 2  # load вызывается для каждого нового store
        expected_add_calls = len(strategies) * 2   # add вызывается для каждого запроса
        expected_save_calls = len(strategies) * 2  # save вызывается после каждого add
        
        assert mock_load.call_count == expected_load_calls
        assert mock_add.call_count == expected_add_calls
        assert mock_save.call_count == expected_save_calls
        
    finally:
        app.dependency_overrides.clear()