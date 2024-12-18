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

from core.vector_store.implementations import PersistentStore, FAISSVectorStore
from core.vector_store.base import VectorStore
from core.vector_store.service import VectorStoreService

logger = logging.getLogger(__name__)

def test_initialization(app_container, store_settings):
    """Test store initialization and directory setup."""
    # Test with non-existent directory
    store_dir = os.path.join(tempfile.mkdtemp(), "new_dir")
    store_settings.faiss_index_path = os.path.join(store_dir, "index.faiss")
    
    # Create directory with proper permissions
    os.makedirs(store_dir, mode=0o755, exist_ok=True)
    
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory
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
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
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
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
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
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
        auto_save=False
    )
    
    store.add(sample_vectors)
    store.save()
    
    # Create backup
    original_path = store_settings.faiss_index_path
    backup_path = original_path + ".backup"
    shutil.copy2(original_path, backup_path)
    
    # Save original file size
    original_size = os.path.getsize(original_path)
    
    # Corrupt original file
    with open(original_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # Reset container to force new store creation
    app_container.reset()
    app_container.configure(store_settings)
    
    # Get new dependencies
    new_base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Try to create new persistent store with corrupted file
    with pytest.raises(RuntimeError):
        store = PersistentStore(
            settings=store_settings,
            store=new_base_store,
            factory=factory
        )
    
    # Restore from backup manually
    shutil.copy2(backup_path, original_path)
    
    # Create new store after recovery
    store = PersistentStore(
        settings=store_settings,
        store=new_base_store,
        factory=factory
    )
    
    assert os.path.getsize(original_path) == original_size

def test_concurrent_operations(app_container, store_settings, sample_vectors):
    """Test concurrent operations with proper synchronization.
    
    FAISS has important thread-safety limitations:
    - Write operations (add) must be synchronized as FAISS doesn't support concurrent writes
    - Read operations (search) can be performed concurrently
    
    This test verifies both scenarios:
    1. Synchronized write operations using threading.Lock
    2. Concurrent read operations which are supported by FAISS
    """
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
        auto_save=False
    )
    
    # Add initial vectors for search testing
    store.add(sample_vectors)
    store.save()
    
    # Test 1: Synchronized write operations
    write_lock = threading.Lock()
    n_write_threads = 5
    
    def add_vectors_safe():
        with write_lock:  # Synchronize FAISS write operations
            store.add(sample_vectors)
            store.save()
    
    # Run concurrent write operations
    write_threads = []
    for _ in range(n_write_threads):
        thread = threading.Thread(target=add_vectors_safe)
        write_threads.append(thread)
        thread.start()
    
    for thread in write_threads:
        thread.join()
    
    # Verify write operations completed successfully
    assert os.path.exists(store_settings.faiss_index_path)
    assert os.path.getsize(store_settings.faiss_index_path) > 0
    expected_vectors = (n_write_threads + 1) * len(sample_vectors)  # +1 for initial vectors
    assert len(store) == expected_vectors
    
    # Test 2: Concurrent read operations
    n_read_threads = 10
    query = sample_vectors[0:1]
    results = []
    
    def search_vectors():
        # Search operations can run concurrently
        distances, indices = store.search(query)
        results.append((distances, indices))
    
    # Run concurrent search operations
    read_threads = []
    for _ in range(n_read_threads):
        thread = threading.Thread(target=search_vectors)
        read_threads.append(thread)
        thread.start()
    
    for thread in read_threads:
        thread.join()
    
    # Verify all search operations returned consistent results
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0][0], results[i][0])  # distances
        np.testing.assert_array_equal(results[0][1], results[i][1])  # indices

@pytest.mark.skipif(os.name == 'nt', reason="Disk space check not supported on Windows")
def test_disk_space_handling(app_container, store_settings, sample_vectors):
    """Test handling of disk space issues."""
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
        auto_save=False
    )
    
    # Mock os.statvfs to simulate disk space issues
    mock_statvfs = MagicMock()
    mock_statvfs.return_value.f_frsize = 4096
    mock_statvfs.return_value.f_bavail = 0  # No free space
    
    with patch('os.statvfs', mock_statvfs):
        with patch.object(store.store, 'save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                store.add(sample_vectors)
                store.save()

def test_store_delegation(app_container, store_settings, sample_vectors):
    """Test basic store delegation."""
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
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
    
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Try to create store with invalid path
    with pytest.raises((OSError, RuntimeError)):
        store = PersistentStore(
            settings=store_settings,
            store=base_store,
            factory=factory
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
                factory=factory
            )
        os.chmod(temp_dir, stat.S_IRWXU)  # Restore permissions for cleanup
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_auto_loading(app_container, store_settings, sample_vectors):
    """Test automatic index loading."""
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create first persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
        auto_save=False
    )
    
    # Add vectors and save
    store.add(sample_vectors)
    store.save()
    
    # Reset container to force new store creation
    app_container.reset()
    app_container.configure(store_settings)
    
    # Get new dependencies
    new_base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create second persistent store with new base store
    new_store = PersistentStore(
        settings=store_settings,
        store=new_base_store,
        factory=factory
    )
    
    # Test search to verify data was loaded
    query = sample_vectors[0:1]
    distances1, indices1 = store.search(query)
    distances2, indices2 = new_store.search(query)
    
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_almost_equal(distances1, distances2)

def test_explicit_loading(app_container, store_settings, sample_vectors):
    """Test explicit index loading."""
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create first persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory,
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
        
        # Get new dependencies
        new_base_store = app_container.get_faiss_store()
        factory = app_container.get_vector_store_factory()
        
        # Create second persistent store with new base store
        new_store = PersistentStore(
            settings=store_settings,
            store=new_base_store,
            factory=factory
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
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=store_settings,
        store=base_store,
        factory=factory
    )
    
    # Test with non-existent file
    with pytest.raises(ValueError):
        store.load("")  # Empty path
        
    with pytest.raises((OSError, RuntimeError)):
        store.load("/nonexistent/path/index.faiss")