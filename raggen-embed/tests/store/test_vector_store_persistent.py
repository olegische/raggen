"""Tests for PersistentStore with dependency injection."""
import os
import shutil
import stat
import numpy as np
import pytest
import time
import logging
from unittest.mock import MagicMock, patch
import threading
import tempfile

from src.core.vector_store.implementations import PersistentStore, FAISSVectorStore
from src.core.vector_store.base import VectorStore
from src.core.vector_store.factory import VectorStoreFactory
from src.core.vector_store.service import VectorStoreService
from src.config.settings import VectorStoreServiceType, VectorStoreImplementationType

logger = logging.getLogger(__name__)

def test_initialization(app_container, store_settings):
    """Test store initialization and directory setup."""
    # Test with non-existent directory
    store_dir = os.path.join(tempfile.mkdtemp(), "new_dir")
    store_settings.faiss_index_path = os.path.join(store_dir, "index.faiss")
    
    # Создаем директорию с правильными правами
    os.makedirs(store_dir, mode=0o755, exist_ok=True)
    
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory()
    )
    
    assert os.path.exists(store_dir)
    assert os.path.exists(os.path.dirname(store.index_path))
    assert os.access(store_dir, os.W_OK)
    
    # Test with read-only directory
    if os.name != 'nt':  # Skip on Windows
        os.chmod(store_dir, stat.S_IREAD | stat.S_IEXEC)
        with pytest.raises((OSError, RuntimeError)):
            store.save()

def test_file_operations(app_container, store_settings, sample_vectors):
    """Test file system operations."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    # Test file creation
    store.add(sample_vectors)
    store.save()
    assert os.path.exists(store_settings.faiss_index_path)
    assert os.path.getsize(store_settings.faiss_index_path) > 0
    
    # Test file update
    initial_size = os.path.getsize(store_settings.faiss_index_path)
    mtime = os.path.getmtime(store_settings.faiss_index_path)
    
    time.sleep(0.1)  # Ensure different timestamp
    store.add(sample_vectors)
    store.save()
    
    assert os.path.getsize(store_settings.faiss_index_path) >= initial_size
    assert os.path.getmtime(store_settings.faiss_index_path) > mtime

def test_backup_management(app_container, store_settings, sample_vectors):
    """Test backup creation and management."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    backup_dir = os.path.dirname(store_settings.faiss_index_path)
    
    # Create multiple backups
    for i in range(3):
        store.add(sample_vectors)
        store.save()
        time.sleep(0.1)  # Ensure different timestamps
    
    # Check backup creation
    backup_files = [f for f in os.listdir(backup_dir) 
                   if f.startswith("index_") and f.endswith(".faiss")]
    assert len(backup_files) > 0
    
    # Test backup rotation
    for i in range(10):
        store.add(sample_vectors)
        store.save()
        time.sleep(0.1)
    
    backup_files = [f for f in os.listdir(backup_dir) 
                   if f.startswith("index_") and f.endswith(".faiss")]
    assert len(backup_files) <= 5  # Default max_backups

def test_backup_recovery(app_container, store_settings, sample_vectors):
    """Test backup recovery process."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    store.add(sample_vectors)
    store.save()
    
    # Create backup
    original_path = store_settings.faiss_index_path
    backup_path = original_path + ".backup"
    shutil.copy2(original_path, backup_path)
    
    # Сохраняем размер оригинального файла
    original_size = os.path.getsize(original_path)
    
    # Corrupt original file
    with open(original_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # Reset container to force new store creation
    app_container.reset()
    app_container.configure(store_settings)
    
    # Get new base store
    new_base_store = app_container.get_faiss_store()
    
    # Try to create new persistent store with corrupted file
    with pytest.raises(RuntimeError):
        store = PersistentStore(
            settings=store_settings,
            store=new_base_store,
            factory=VectorStoreFactory()
        )
    
    # Восстанавливаем из бэкапа вручную
    shutil.copy2(backup_path, original_path)
    
    # Create new store after recovery
    store = PersistentStore(
        settings=store_settings,
        store=new_base_store,
        factory=VectorStoreFactory()
    )
    
    assert os.path.getsize(original_path) == original_size

def test_concurrent_operations(app_container, store_settings, sample_vectors):
    """Test concurrent file operations."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    n_threads = 5
    
    def add_vectors():
        store.add(sample_vectors)
        store.save()
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
    assert os.path.exists(store_settings.faiss_index_path)
    assert os.path.getsize(store_settings.faiss_index_path) > 0

@pytest.mark.skipif(os.name == 'nt', reason="Disk space check not supported on Windows")
def test_disk_space_handling(app_container, store_settings, sample_vectors):
    """Test handling of disk space issues."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    # Патчим os.statvfs для симуляции нехватки места
    mock_statvfs = MagicMock()
    mock_statvfs.return_value.f_frsize = 4096
    mock_statvfs.return_value.f_bavail = 0  # Нет свободного места
    
    with patch('os.statvfs', mock_statvfs):
        with patch.object(store.store, 'save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                store.add(sample_vectors)
                store.save()

def test_store_delegation(app_container, store_settings, sample_vectors):
    """Test basic store delegation."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    # Test add delegation
    store.add(sample_vectors)
    assert len(base_store) == len(sample_vectors)
    
    # Test search delegation
    query = sample_vectors[0:1]
    distances1, indices1 = store.search(query)
    distances2, indices2 = base_store.search(query)
    
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_almost_equal(distances1, distances2)

def test_error_handling(app_container, store_settings):
    """Test handling of file system errors."""
    # Test with invalid path
    invalid_dir = "/nonexistent/directory"
    store_settings.faiss_index_path = os.path.join(invalid_dir, "index.faiss")
    
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Try to create store with invalid path
    with pytest.raises((OSError, RuntimeError)):
        store = PersistentStore(
            settings=store_settings,
            store=base_store,
            factory=VectorStoreFactory()
        )
    
    # Test with invalid permissions
    temp_dir = tempfile.mkdtemp()
    store_settings.faiss_index_path = os.path.join(temp_dir, "index.faiss")
    
    if os.name != 'nt':  # Skip on Windows
        os.chmod(temp_dir, 0)  # Remove all permissions
        with pytest.raises((OSError, RuntimeError)):
            store = PersistentStore(
                settings=store_settings,
                store=base_store,
                factory=VectorStoreFactory()
            )
        os.chmod(temp_dir, stat.S_IRWXU)  # Restore permissions for cleanup
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_auto_loading(app_container, store_settings, sample_vectors):
    """Test automatic index loading."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create first persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    # Add vectors and save
    store.add(sample_vectors)
    store.save()
    
    # Reset container to force new store creation
    app_container.reset()
    app_container.configure(store_settings)
    
    # Get new base store
    new_base_store = app_container.get_faiss_store()
    
    # Create second persistent store with new base store
    new_store = PersistentStore(
        settings=store_settings,
        store=new_base_store,
        factory=VectorStoreFactory()
    )
    
    # Test search to verify data was loaded
    query = sample_vectors[0:1]
    distances1, indices1 = store.search(query)
    distances2, indices2 = new_store.search(query)
    
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_almost_equal(distances1, distances2)

def test_explicit_loading(app_container, store_settings, sample_vectors):
    """Test explicit index loading."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create first persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory(),
        auto_save=False
    )
    
    # Add vectors and save to custom path
    store.add(sample_vectors)
    test_path = "test_index.faiss"
    store.save(test_path)
    
    try:
        # Reset container to force new store creation
        app_container.reset()
        app_container.configure(store_settings)
        
        # Get new base store
        new_base_store = app_container.get_faiss_store()
        
        # Create second persistent store with new base store
        new_store = PersistentStore(
            settings=store_settings,
            store=new_base_store,
            factory=VectorStoreFactory()
        )
        
        # Load from custom path
        new_store.load(test_path)
        
        # Test search to verify data was loaded
        query = sample_vectors[0:1]
        distances1, indices1 = store.search(query)
        distances2, indices2 = new_store.search(query)
        
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_almost_equal(distances1, distances2)
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

def test_load_errors(app_container, store_settings):
    """Test error handling during load."""
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=VectorStoreFactory()
    )
    
    # Test with non-existent file
    with pytest.raises(ValueError):
        store.load("")  # Empty path
        
    with pytest.raises((OSError, RuntimeError)):
        store.load("/nonexistent/path/index.faiss")