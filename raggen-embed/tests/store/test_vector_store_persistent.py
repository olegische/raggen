import os
import shutil
import stat
import numpy as np
import pytest
from datetime import datetime
import time
import logging
from unittest.mock import MagicMock, patch, Mock
import threading
import tempfile

from core.vector_store.implementations import PersistentStore, FAISSVectorStore
from core.vector_store.base import VectorStore
from core.vector_store.factory import VectorStoreFactory
from core.vector_store.service import VectorStoreService
from config.settings import Settings, reset_settings, VectorStoreServiceType, VectorStoreImplementationType

logger = logging.getLogger(__name__)

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
    
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    assert os.path.exists(store_dir)
    assert os.path.exists(os.path.dirname(store.index_path))
    assert os.access(store_dir, os.W_OK)
    
    # Test with existing directory
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    assert os.path.exists(store_dir)
    
    # Test with read-only directory
    if os.name != 'nt':  # Skip on Windows
        os.chmod(store_dir, stat.S_IREAD | stat.S_IEXEC)
        with pytest.raises((OSError, RuntimeError)):
            store.save()

def test_file_operations(test_settings, test_vectors):
    """Test file system operations."""
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    
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
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
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
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
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
        store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    
    # Восстанавливаем из бэкапа вручную
    shutil.copy2(backup_path, original_path)
    
    # Проверяем восстановление
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    assert os.path.getsize(original_path) == original_size

def test_concurrent_operations(test_settings, test_vectors):
    """Test concurrent file operations."""
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
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
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    
    # Патчим os.statvfs для симуляции нехватки места
    mock_statvfs = MagicMock()
    mock_statvfs.return_value.f_frsize = 4096
    mock_statvfs.return_value.f_bavail = 0  # Нет свободного места
    
    with patch('os.statvfs', mock_statvfs):
        with patch.object(store.store, 'save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                store.add(test_vectors)

def test_store_delegation(mock_vector_store, mock_vector_store_factory, test_settings, test_vectors):
    """Test basic store delegation."""
    # Устанавливаем тип реализации в настройках
    test_settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    
    # Create persistent store with mocked dependencies
    store = PersistentStore(
        settings=test_settings,
        factory=mock_vector_store_factory,
        auto_save=False  # Отключаем автосохранение для теста
    )
    
    # Test add delegation
    store.add(test_vectors)
    mock_vector_store.add.assert_called_once()
    
    # Test search delegation
    query = np.random.randn(1, 384).astype(np.float32)
    store.search(query)
    mock_vector_store.search.assert_called_once()
    
    # Test len delegation
    len(store)
    mock_vector_store.__len__.assert_called_once()

def test_error_handling(test_settings):
    """Test handling of file system errors."""
    # Test with invalid path
    invalid_dir = "/nonexistent/directory"
    test_settings.faiss_index_path = os.path.join(invalid_dir, "index.faiss")
    
    factory = VectorStoreFactory()
    with pytest.raises((OSError, RuntimeError)):
        store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    
    # Test with invalid permissions
    temp_dir = tempfile.mkdtemp()
    test_settings.faiss_index_path = os.path.join(temp_dir, "index.faiss")
    
    if os.name != 'nt':  # Skip on Windows
        os.chmod(temp_dir, 0)  # Remove all permissions
        with pytest.raises((OSError, RuntimeError)):
            store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
        os.chmod(temp_dir, stat.S_IRWXU)  # Restore permissions for cleanup
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_auto_loading(test_settings, test_vectors):
    """Test automatic index loading through VectorStoreService."""
    # Initialize service
    factory = VectorStoreFactory()
    vector_store_service = VectorStoreService(test_settings, factory)
    
    # Add vectors to store
    store = vector_store_service.store
    store.add(test_vectors)
    
    logger.info("Initial vectors added.")
    
    # Ensure file is saved
    assert os.path.exists(test_settings.faiss_index_path)
    assert os.path.getsize(test_settings.faiss_index_path) > 0
    
    logger.info("Files saved.")
    
    # Reset service to force reload
    vector_store_service.reset()
    reloaded_store = vector_store_service.store
    
    # Test search to verify data was loaded
    query = np.random.randn(1, 384).astype(np.float32)
    distances1, indices1 = store.search(query)
    logger.info("First search for data completed")
    distances2, indices2 = reloaded_store.search(query)
    logger.info("Second search for data completed")
    
    assert isinstance(distances1, np.ndarray)
    assert isinstance(distances2, np.ndarray)
    np.testing.assert_array_equal(indices1, indices2)

def test_explicit_loading(test_settings, test_vectors):
    """Test explicit index loading."""
    factory = VectorStoreFactory()
    
    # Create and save initial index
    store1 = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    store1.add(test_vectors)
    
    # Create new store and load index from different path
    test_path = "test_index.faiss"
    store1.save(test_path)
    
    try:
        # Create new store and load index explicitly
        store2 = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
        store2.load(test_path)
        
        # Test search to verify data was loaded
        query = np.random.randn(1, 384).astype(np.float32)
        distances1, indices1 = store1.search(query)
        logger.info("First search for data completed")
        distances2, indices2 = store2.search(query)
        logger.info("Second search for data completed")
        
        assert isinstance(distances1, np.ndarray)
        assert isinstance(distances2, np.ndarray)
        np.testing.assert_array_equal(indices1, indices2)
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

def test_load_errors(test_settings):
    """Test error handling during load."""
    factory = VectorStoreFactory()
    store = factory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    
    # Test with non-existent file
    with pytest.raises(ValueError):
        store.load("")  # Empty path
        
    with pytest.raises((OSError, RuntimeError)):
        store.load("/nonexistent/path/index.faiss")

def test_factory_creates_persistent_store(test_settings):
    """Test that factory creates PersistentStore correctly."""
    store = VectorStoreFactory.create(VectorStoreServiceType.PERSISTENT, test_settings)
    assert isinstance(store, PersistentStore)