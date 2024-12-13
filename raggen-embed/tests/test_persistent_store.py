import os
import shutil
import numpy as np
import pytest
from datetime import datetime, timedelta
import time

from core.vector_store.persistent_store import PersistentFAISSStore
from config.settings import Settings

settings = Settings()

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "data" / "faiss"
    os.makedirs(test_dir, exist_ok=True)
    
    # Patch settings to use test directory
    original_path = settings.faiss_index_path
    settings.faiss_index_path = str(test_dir / "index.faiss")
    
    yield test_dir
    
    # Restore original path
    settings.faiss_index_path = original_path
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

def test_initialization(test_dir):
    """Test store initialization."""
    store = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    assert len(store) == 0
    assert os.path.exists(test_dir)

def test_persistence(test_dir, sample_vectors):
    """Test that vectors persist between store instances."""
    # Create and add vectors to first instance
    store1 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    store1.add_vectors(sample_vectors)
    initial_count = len(store1)

    # Create second instance and verify vectors are loaded
    store2 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == initial_count

    # Test search functionality
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances1, indices1 = store1.search(query)
    distances2, indices2 = store2.search(query)
    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_almost_equal(distances1, distances2)

def test_backup_creation(test_dir, sample_vectors):
    """Test backup file creation."""
    store = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    
    # Add vectors multiple times to trigger backups
    for i in range(3):
        store.add_vectors(sample_vectors[i*1000:(i+1)*1000])
        time.sleep(1)  # Ensure different timestamps
    
    # Check that backup files exist
    backup_files = [
        f for f in os.listdir(test_dir)
        if f.startswith("index_") and f.endswith(".faiss")
    ]
    assert len(backup_files) > 0

def test_backup_cleanup(test_dir, sample_vectors):
    """Test cleanup of old backup files."""
    store = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    
    # Add vectors multiple times to create many backups
    for i in range(10):
        store.add_vectors(sample_vectors[i*1000:(i+1)*1000])
        time.sleep(1)  # Ensure different timestamps
    
    # Check that only the specified number of backups are kept
    backup_files = [
        f for f in os.listdir(test_dir)
        if f.startswith("index_") and f.endswith(".faiss")
    ]
    assert len(backup_files) <= 5  # Default is to keep last 5

def test_dimension_mismatch_warning(test_dir, sample_vectors):
    """Test warning when loading index with different dimensions."""
    # Create initial store
    store1 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"), dimension=384)
    store1.add_vectors(sample_vectors)

    # Try to create store with different dimension
    with pytest.warns(UserWarning, match="Loaded index dimension .* differs from requested"):
        store2 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"), dimension=512)
        assert store2.store.dimension == 384  # Should use loaded dimension

def test_auto_save_disabled(test_dir, sample_vectors):
    """Test behavior when auto_save is disabled."""
    # Create store with auto_save disabled
    store1 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"), auto_save=False)
    store1.add_vectors(sample_vectors)

    # Create new instance - should not have the vectors since auto_save is disabled
    store2 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == 0

def test_backup_restoration(test_dir, sample_vectors, monkeypatch):
    """Test backup restoration after failed save."""
    store = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    store.add_vectors(sample_vectors[:5000])  # Add initial vectors

    # Mock faiss.write_index to fail
    def mock_save(*args):
        raise RuntimeError("Simulated save failure")

    # Add more vectors with mocked save
    with monkeypatch.context() as m:
        m.setattr("faiss.write_index", mock_save)
        with pytest.raises(RuntimeError, match="Simulated save failure"):
            store.add_vectors(sample_vectors[5000:])  # Should fail to save but restore backup

    # Create new instance and verify it has the initial vectors
    store2 = PersistentFAISSStore(index_path=str(test_dir / "index.faiss"))
    assert len(store2) == 5000  # Should have only the initial vectors