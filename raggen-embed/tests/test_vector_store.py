import os
import tempfile
import shutil
import signal
from contextlib import contextmanager
import numpy as np
import pytest
import time

from core.vector_store.faiss_store import FAISSVectorStore
from config.settings import Settings

settings = Settings()

@pytest.fixture
def vector_store():
    """Fixture for vector store."""
    return FAISSVectorStore()

@pytest.fixture
def sample_vectors():
    """Fixture for sample vectors."""
    # Generate random vectors for testing
    # FAISS IVF requires at least 39 * n_clusters points for training
    n_vectors = 39 * settings.n_clusters + 100  # Add some extra vectors
    return np.random.randn(n_vectors, settings.vector_dim).astype(np.float32)

@pytest.fixture
def large_vectors():
    """Fixture for large vector dataset."""
    # Generate 100K vectors
    n_vectors = 100_000
    return np.random.randn(n_vectors, settings.vector_dim).astype(np.float32)

@contextmanager
def timeout(seconds):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def test_initialization():
    """Test vector store initialization."""
    store = FAISSVectorStore(dimension=384)
    assert store.dimension == 384
    assert store.n_vectors == 0
    assert not store.is_trained

def test_training(vector_store, sample_vectors):
    """Test index training."""
    vector_store.train(sample_vectors)
    assert vector_store.is_trained
    
    # Test training already trained index
    vector_store.train(sample_vectors)  # Should just log warning

def test_adding_vectors(vector_store, sample_vectors):
    """Test adding vectors to the index."""
    # Train first
    vector_store.train(sample_vectors)
    
    # Test adding vectors without IDs
    n_add = 10
    vectors_to_add = np.random.randn(n_add, settings.vector_dim).astype(np.float32)
    ids = vector_store.add_vectors(vectors_to_add)
    assert len(ids) == n_add
    assert vector_store.n_vectors == n_add
    
    # Test adding vectors with IDs
    custom_ids = np.array([100, 101, 102])
    vectors_with_ids = np.random.randn(3, settings.vector_dim).astype(np.float32)
    ids = vector_store.add_vectors(vectors_with_ids, custom_ids)
    assert ids == custom_ids.tolist()
    assert vector_store.n_vectors == n_add + 3
    
    # Test adding vectors with wrong dimension
    wrong_dim_vectors = np.random.randn(5, settings.vector_dim + 1).astype(np.float32)
    with pytest.raises(ValueError, match="Expected vectors of dimension"):
        vector_store.add_vectors(wrong_dim_vectors)
    
    # Test adding vectors without training
    new_store = FAISSVectorStore()
    with pytest.raises(RuntimeError, match="Index must be trained"):
        new_store.add_vectors(vectors_to_add)

def test_searching(vector_store, sample_vectors):
    """Test vector similarity search."""
    # Train and add vectors
    vector_store.train(sample_vectors)
    vector_store.add_vectors(sample_vectors)
    
    # Test basic search
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances, indices = vector_store.search(query, k=5)
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    # Test batch search
    queries = np.random.randn(3, settings.vector_dim).astype(np.float32)
    distances, indices = vector_store.search(queries, k=5)
    assert distances.shape == (3, 5)
    assert indices.shape == (3, 5)
    
    # Test search with wrong dimension
    wrong_query = np.random.randn(1, settings.vector_dim + 1).astype(np.float32)
    with pytest.raises(ValueError, match="Expected vectors of dimension"):
        vector_store.search(wrong_query)
    
    # Test search without training
    new_store = FAISSVectorStore()
    with pytest.raises(RuntimeError, match="Index must be trained"):
        new_store.search(query)
    
    # Test search with k > n_vectors
    k_too_large = len(sample_vectors) + 10
    distances, indices = vector_store.search(query, k=k_too_large)
    assert distances.shape == (1, vector_store.n_vectors)
    assert indices.shape == (1, vector_store.n_vectors)

def test_persistence(vector_store, sample_vectors):
    """Test saving and loading the index."""
    # Train and add vectors
    vector_store.train(sample_vectors)
    vector_store.add_vectors(sample_vectors)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        try:
            # Save index
            vector_store.save(tmp.name)
            assert os.path.exists(tmp.name)
            
            # Load index
            loaded_store = FAISSVectorStore.load(tmp.name)
            assert loaded_store.dimension == vector_store.dimension
            assert loaded_store.n_vectors == vector_store.n_vectors
            assert loaded_store.is_trained
            
            # Test search with loaded index
            query = np.random.randn(1, settings.vector_dim).astype(np.float32)
            original_distances, original_indices = vector_store.search(query)
            loaded_distances, loaded_indices = loaded_store.search(query)
            np.testing.assert_array_equal(original_indices, loaded_indices)
            np.testing.assert_array_almost_equal(original_distances, loaded_distances)
            
        finally:
            # Cleanup
            os.unlink(tmp.name)
    
    # Test loading non-existent file
    with pytest.raises(Exception):
        FAISSVectorStore.load("non_existent_file.faiss")

def test_large_dataset(vector_store, large_vectors):
    """Test handling of large datasets."""
    # Train the index
    vector_store.train(large_vectors[:10000])  # Use first 10K vectors for training
    assert vector_store.is_trained
    
    # Add vectors in batches
    batch_size = 10000
    n_batches = len(large_vectors) // batch_size
    
    start_time = time.time()
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = large_vectors[start_idx:end_idx]
        vector_store.add_vectors(batch)
        
    total_time = time.time() - start_time
    vectors_per_second = len(large_vectors) / total_time
    
    # Log performance metrics
    print(f"\nLarge dataset performance:")
    print(f"Total vectors: {len(large_vectors):,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Vectors per second: {vectors_per_second:.2f}")
    
    assert len(vector_store) == len(large_vectors)
    
    # Test search performance
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    with timeout(1):  # Search should complete within 1 second
        distances, indices = vector_store.search(query, k=10)
    assert len(indices[0]) == 10

def test_persistence_large_index(vector_store, large_vectors, tmp_path):
    """Test persistence with large index."""
    # Prepare large index
    vector_store.train(large_vectors[:10000])
    vector_store.add_vectors(large_vectors)
    
    # Save index
    index_path = tmp_path / "large_index.faiss"
    start_time = time.time()
    vector_store.save(str(index_path))
    save_time = time.time() - start_time
    
    # Get file size
    index_size = os.path.getsize(index_path) / (1024 * 1024)  # Size in MB
    
    # Load index
    start_time = time.time()
    loaded_store = FAISSVectorStore.load(str(index_path))
    load_time = time.time() - start_time
    
    # Log persistence metrics
    print(f"\nPersistence metrics:")
    print(f"Index size: {index_size:.2f}MB")
    print(f"Save time: {save_time:.2f}s")
    print(f"Load time: {load_time:.2f}s")
    
    # Verify loaded index
    assert loaded_store.n_vectors == len(large_vectors)
    assert loaded_store.dimension == settings.vector_dim
    
    # Test search with loaded index
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances1, indices1 = vector_store.search(query)
    distances2, indices2 = loaded_store.search(query)
    np.testing.assert_array_equal(indices1, indices2)

def test_recovery(vector_store, large_vectors, tmp_path):
    """Test index recovery after simulated failures."""
    # Prepare data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Function to simulate crash during save
    def simulate_crash_save(store, vectors, save_path):
        store.train(vectors[:10000])
        store.add_vectors(vectors[:50000])  # Add only half the vectors
        store.save(save_path)  # Save partial index
        
    # Function to simulate crash during add
    def simulate_crash_add(store, vectors):
        store.train(vectors[:10000])
        try:
            # Simulate crash by adding vectors with wrong dimension
            bad_vectors = np.random.randn(100, settings.vector_dim + 1).astype(np.float32)
            store.add_vectors(bad_vectors)
        except ValueError:
            pass  # Expected error
    
    # Test recovery from partial save
    partial_path = str(data_dir / "partial.faiss")
    simulate_crash_save(vector_store, large_vectors, partial_path)
    
    # Load partial index and complete it
    recovered_store = FAISSVectorStore.load(partial_path)
    assert recovered_store.n_vectors == 50000  # Should have half the vectors
    
    # Add remaining vectors
    recovered_store.add_vectors(large_vectors[50000:])
    assert recovered_store.n_vectors == len(large_vectors)
    
    # Test recovery from failed add
    new_store = FAISSVectorStore()
    simulate_crash_add(new_store, large_vectors)
    
    # Should be able to continue after error
    new_store.add_vectors(large_vectors[10000:])
    assert new_store.n_vectors == len(large_vectors) - 10000
    
    # Verify search still works
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances, indices = new_store.search(query)
    assert len(indices[0]) == settings.n_results
