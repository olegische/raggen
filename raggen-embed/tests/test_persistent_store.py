import os
import shutil
import numpy as np
import pytest
from datetime import datetime, timedelta
import time

from core.vector_store.persistent_store import PersistentStore
from core.vector_store.faiss_store import FAISSVectorStore
from config.settings import Settings

settings = Settings()

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "data" / "faiss"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir.parent)

@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    # Generate enough vectors for any index type
    n_vectors = max(
        39 * settings.n_clusters,  # For IVF_FLAT
        39 * 256,  # For IVF_PQ (default n_centroids)
        10000  # Minimum reasonable size
    ) + 100  # Add some extra vectors
    return np.random.randn(n_vectors, settings.vector_dim).astype(np.float32)

@pytest.fixture
def vector_store():
    """Create a FAISSVectorStore instance."""
    return FAISSVectorStore()

def test_initialization(test_dir):
    """Test store initialization."""
    # Test with default store
    store = PersistentStore(
        index_path=str(test_dir / "index.faiss")
    )
    assert len(store) == 0
    assert os.path.exists(test_dir)
    assert isinstance(store.store, FAISSVectorStore)

    # Test with injected store
    custom_store = FAISSVectorStore(dimension=512)
    store = PersistentStore(
        store=custom_store,
        index_path=str(test_dir / "custom_index.faiss")
    )
    assert len(store) == 0
    assert store.store.dimension == 512

def test_persistence(test_dir, sample_vectors):
    """Test that vectors persist between store instances."""
    # Create and add vectors to first instance
    store1 = PersistentStore(index_path=str(test_dir / "index.faiss"))
    store1.add(sample_vectors)
    initial_count = len(store1)

    # Create second instance and verify vectors are loaded
    store2 = PersistentStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == initial_count

    # Test search functionality
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances1, indices1 = store1.search(query)
    distances2, indices2 = store2.search(query)
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_almost_equal(distances1, distances2)

def test_backup_creation(test_dir, sample_vectors):
    """Test backup file creation."""
    store = PersistentStore(index_path=str(test_dir / "index.faiss"))
    
    # Add vectors multiple times to trigger backups
    for i in range(3):
        store.add(sample_vectors[i*1000:(i+1)*1000])
        time.sleep(1)  # Ensure different timestamps
    
    # Check that backup files exist
    backup_files = [
        f for f in os.listdir(test_dir)
        if f.startswith("index_") and f.endswith(".faiss")
    ]
    assert len(backup_files) > 0

def test_backup_cleanup(test_dir, sample_vectors):
    """Test cleanup of old backup files."""
    store = PersistentStore(index_path=str(test_dir / "index.faiss"))
    
    # Add vectors multiple times to create many backups
    for i in range(10):
        store.add(sample_vectors[i*1000:(i+1)*1000])
        time.sleep(1)  # Ensure different timestamps
    
    # Check that only the specified number of backups are kept
    backup_files = [
        f for f in os.listdir(test_dir)
        if f.startswith("index_") and f.endswith(".faiss")
    ]
    assert len(backup_files) <= 5  # Default is to keep last 5

def test_auto_save_disabled(test_dir, sample_vectors):
    """Test behavior when auto_save is disabled."""
    # Create store with auto_save disabled
    store1 = PersistentStore(
        index_path=str(test_dir / "index.faiss"),
        auto_save=False
    )
    store1.add(sample_vectors)

    # Create new instance - should not have the vectors since auto_save is disabled
    store2 = PersistentStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == 0

def test_backup_restoration(test_dir, sample_vectors, monkeypatch):
    """Test backup restoration after failed save."""
    store = PersistentStore(index_path=str(test_dir / "index.faiss"))
    store.add(sample_vectors[:5000])  # Add initial vectors

    # Mock faiss.write_index to fail
    def mock_save(*args):
        raise RuntimeError("Simulated save failure")

    # Add more vectors with mocked save
    with monkeypatch.context() as m:
        m.setattr("faiss.write_index", mock_save)
        with pytest.raises(RuntimeError, match="Simulated save failure"):
            store.add(sample_vectors[5000:])  # Should fail to save but restore backup

    # Create new instance and verify it has the initial vectors
    store2 = PersistentStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == 5000  # Should have only the initial vectors

def test_custom_store_persistence(test_dir, sample_vectors):
    """Test persistence with custom injected store."""
    # Create store with custom dimension
    custom_store = FAISSVectorStore(dimension=512)
    store1 = PersistentStore(
        store=custom_store,
        index_path=str(test_dir / "index.faiss")
    )
    
    # Create vectors with custom dimension
    custom_vectors = np.random.randn(1000, 512).astype(np.float32)
    store1.add(custom_vectors)
    
    # Load in new instance and verify dimension
    store2 = PersistentStore(index_path=str(test_dir / "index.faiss"))
    assert store2.store.dimension == 512
    assert len(store2) == len(custom_vectors)
    
    # Verify search works
    query = np.random.randn(1, 512).astype(np.float32)
    distances1, indices1 = store1.search(query)
    distances2, indices2 = store2.search(query)
    np.testing.assert_array_equal(indices1, indices2)