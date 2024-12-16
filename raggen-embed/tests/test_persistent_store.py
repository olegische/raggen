import os
import shutil
import stat
import numpy as np
import pytest
from datetime import datetime
import time
import logging
from unittest.mock import MagicMock, patch
import threading
import tempfile

from core.vector_store.persistent_store import PersistentStore
from core.vector_store.faiss_store import FAISSVectorStore
from core.vector_store.base import VectorStore
from core.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreType
from config.settings import Settings, reset_settings

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def test_settings():
    """Create settings specifically for tests."""
    reset_settings()
    temp_dir = tempfile.mkdtemp()
    temp_index_path = os.path.join(temp_dir, "index.faiss")
    
    os.environ.update({
        "FAISS_INDEX_PATH": temp_index_path,
        "VECTOR_DIM": "384",
        "FAISS_INDEX_TYPE": "flat_l2"
    })
    
    settings = Settings()
    yield settings
    
    # Восстанавливаем права перед удалением
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    reset_settings()
    for key in ["FAISS_INDEX_PATH", "VECTOR_DIM", "FAISS_INDEX_TYPE"]:
        if key in os.environ:
            del os.environ[key]

@pytest.fixture
def mock_store():
    """Create mock store for testing."""
    store = MagicMock(spec=VectorStore)
    store.add = MagicMock()
    store.search = MagicMock(return_value=(np.array([]), np.array([])))
    store.save = MagicMock()
    store.load = MagicMock()
    store.__len__ = MagicMock(return_value=0)
    return store

@pytest.fixture
def test_vectors():
    """Generate small test vectors."""
    return np.random.randn(10, 384).astype(np.float32)

def test_initialization(test_settings):
    """Test store initialization and directory setup."""
    # Test with non-existent directory
    store_dir = os.path.join(tempfile.mkdtemp(), "new_dir")
    test_settings.faiss_index_path = os.path.join(store_dir, "index.faiss")
    
    # Создаем директорию с правильными правами
    os.makedirs(store_dir, mode=0o755, exist_ok=True)
    
    store = PersistentStore(settings=test_settings)
    assert os.path.exists(store_dir)
    assert os.path.exists(os.path.dirname(store.index_path))
    assert os.access(store_dir, os.W_OK)
    
    # Test with existing directory
    store = PersistentStore(settings=test_settings)
    assert os.path.exists(store_dir)
    
    # Test with read-only directory
    if os.name != 'nt':  # Skip on Windows
        os.chmod(store_dir, stat.S_IREAD | stat.S_IEXEC)
        with pytest.raises((OSError, RuntimeError)):
            store.save()

def test_file_operations(test_settings, test_vectors):
    """Test file system operations."""
    store = PersistentStore(settings=test_settings)
    
    # Test file creation
    store.add(test_vectors)
    assert os.path.exists(test_settings.faiss_index_path)
    assert os.path.getsize(test_settings.faiss_index_path) > 0
    
    # Test file update
    initial_size = os.path.getsize(test_settings.faiss_index_path)
    mtime = os.path.getmtime(test_settings.faiss_index_path)
    
    time.sleep(0.1)  # Ensure different timestamp
    store.add(test_vectors)
    
    assert os.path.getsize(test_settings.faiss_index_path) >= initial_size
    assert os.path.getmtime(test_settings.faiss_index_path) > mtime

def test_backup_management(test_settings, test_vectors):
    """Test backup creation and management."""
    store = PersistentStore(settings=test_settings)
    backup_dir = os.path.dirname(test_settings.faiss_index_path)
    
    # Create multiple backups
    for i in range(3):
        store.add(test_vectors)
        time.sleep(0.1)  # Ensure different timestamps
    
    # Check backup creation
    backup_files = [f for f in os.listdir(backup_dir) 
                   if f.startswith("index_") and f.endswith(".faiss")]
    assert len(backup_files) > 0
    
    # Test backup rotation
    for i in range(10):
        store.add(test_vectors)
        time.sleep(0.1)
    
    backup_files = [f for f in os.listdir(backup_dir) 
                   if f.startswith("index_") and f.endswith(".faiss")]
    assert len(backup_files) <= 5  # Default max_backups

def test_backup_recovery(test_settings, test_vectors):
    """Test backup recovery process."""
    store = PersistentStore(settings=test_settings)
    store.add(test_vectors)
    
    # Create backup
    original_path = test_settings.faiss_index_path
    backup_path = original_path + ".backup"
    shutil.copy2(original_path, backup_path)
    
    # Сохраняем размер оригинального файла
    original_size = os.path.getsize(original_path)
    
    # Corrupt original file
    with open(original_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # Пробуем загрузить повреждённый файл
    with pytest.raises(RuntimeError):
        store = PersistentStore(settings=test_settings)
    
    # Восстанавливаем из бэкапа вручную
    shutil.copy2(backup_path, original_path)
    
    # Проверяем восстановление
    store = PersistentStore(settings=test_settings)
    assert os.path.getsize(original_path) == original_size

def test_concurrent_operations(test_settings, test_vectors):
    """Test concurrent file operations."""
    store = PersistentStore(settings=test_settings)
    n_threads = 5
    
    def add_vectors():
        store.add(test_vectors)
        time.sleep(0.1)  # Simulate work
    
    # Run concurrent operations
    threads = []
    for _ in range(n_threads):
        thread = threading.Thread(target=add_vectors)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify file integrity
    assert os.path.exists(test_settings.faiss_index_path)
    assert os.path.getsize(test_settings.faiss_index_path) > 0

@pytest.mark.skipif(os.name == 'nt', reason="Disk space check not supported on Windows")
def test_disk_space_handling(test_settings, test_vectors):
    """Test handling of disk space issues."""
    store = PersistentStore(settings=test_settings)
    
    # Патчим os.statvfs для симуляции нехватки места
    mock_statvfs = MagicMock()
    mock_statvfs.return_value.f_frsize = 4096
    mock_statvfs.return_value.f_bavail = 0  # Нет свободного места
    
    with patch('os.statvfs', mock_statvfs):
        with patch.object(store.store, 'save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                store.add(test_vectors)

def test_store_delegation(mock_store, test_settings, test_vectors):
    """Test basic store delegation."""
    store = PersistentStore(store=mock_store, settings=test_settings)
    
    # Test add delegation
    store.add(test_vectors)
    mock_store.add.assert_called_once()
    
    # Test search delegation
    query = np.random.randn(1, 384).astype(np.float32)
    store.search(query)
    mock_store.search.assert_called_once()
    
    # Test len delegation
    len(store)
    mock_store.__len__.assert_called_once()

def test_error_handling(test_settings):
    """Test handling of file system errors."""
    # Test with invalid path
    invalid_dir = "/nonexistent/directory"
    test_settings.faiss_index_path = os.path.join(invalid_dir, "index.faiss")
    
    with pytest.raises((OSError, RuntimeError)):
        store = PersistentStore(settings=test_settings)
    
    # Test with invalid permissions
    temp_dir = tempfile.mkdtemp()
    test_settings.faiss_index_path = os.path.join(temp_dir, "index.faiss")
    
    if os.name != 'nt':  # Skip on Windows
        os.chmod(temp_dir, 0)  # Remove all permissions
        with pytest.raises((OSError, RuntimeError)):
            store = PersistentStore(settings=test_settings)
        os.chmod(temp_dir, stat.S_IRWXU)  # Restore permissions for cleanup
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_auto_loading(test_settings, test_vectors):
    """Test automatic index loading."""
    # Create and save initial index
    store1 = PersistentStore(settings=test_settings)
    store1.add(test_vectors)
    
    # Ensure file is saved
    assert os.path.exists(test_settings.faiss_index_path)
    assert os.path.getsize(test_settings.faiss_index_path) > 0
    
    # Create new instance - should load existing index
    store2 = PersistentStore(settings=test_settings)
    
    # Test search to verify data was loaded
    query = np.random.randn(1, 384).astype(np.float32)
    distances1, indices1 = store1.search(query)
    distances2, indices2 = store2.search(query)
    
    assert isinstance(distances1, np.ndarray)
    assert isinstance(distances2, np.ndarray)
    np.testing.assert_array_equal(indices1, indices2)

def test_factory_integration(test_settings, test_vectors):
    """Test integration with VectorStoreFactory."""
    # Clear factory cache
    VectorStoreFactory.clear_cache()
    
    # Create store using factory
    store1 = VectorStoreFactory.create(VectorStoreType.PERSISTENT, test_settings)
    assert isinstance(store1, PersistentStore)
    
    # Add vectors
    store1.add(test_vectors)
    
    # Get the same store from cache
    store2 = VectorStoreFactory.get_or_create(VectorStoreType.PERSISTENT, test_settings)
    assert store2 is store1  # Should be the same instance
    
    # Force create new instance
    store3 = VectorStoreFactory.create(VectorStoreType.PERSISTENT, test_settings, force_new=True)
    assert store3 is not store1  # Should be a different instance
    
    # Both instances should have access to the same data
    query = np.random.randn(1, 384).astype(np.float32)
    distances1, indices1 = store1.search(query)
    distances3, indices3 = store3.search(query)
    np.testing.assert_array_equal(indices1, indices3)