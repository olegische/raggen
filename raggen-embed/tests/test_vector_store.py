import os
import tempfile
import shutil
import signal
from contextlib import contextmanager
import numpy as np
import pytest
import time
import faiss

from core.vector_store.faiss_store import FAISSVectorStore
from config.settings import Settings, IndexType

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
    # For IVF_PQ we need at least 39 * 256 = 9984 points
    n_vectors = max(
        39 * settings.n_clusters,  # For IVF_FLAT
        39 * 256,  # For IVF_PQ (default n_centroids)
        10000  # Minimum reasonable size
    ) + 100  # Add some extra vectors
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

def test_adding_vectors(vector_store, sample_vectors):
    """Test adding vectors to the index."""
    # Add vectors in batches to test automatic training
    n_add = 10
    vectors_to_add = np.random.randn(n_add, settings.vector_dim).astype(np.float32)
    vector_store.add(vectors_to_add)
    assert vector_store.n_vectors == n_add
    assert vector_store.is_trained  # Should be trained after first add
    
    # Add more vectors
    more_vectors = np.random.randn(5, settings.vector_dim).astype(np.float32)
    vector_store.add(more_vectors)
    assert vector_store.n_vectors == n_add + 5
    
    # Test that search works after adding vectors
    query = vectors_to_add[0:1]  # Use first vector as query
    distances, indices = vector_store.search(query, k=1)
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)

def test_searching(vector_store, sample_vectors):
    """Test vector similarity search."""
    # Add vectors (should handle training automatically)
    vector_store.add(sample_vectors)
    
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
    # Add vectors (should handle training automatically)
    vector_store.add(sample_vectors)

    # Create test file path in the current directory
    test_path = "test_index.faiss"
    try:
        # Save index
        vector_store.save(test_path)
        assert os.path.exists(test_path), "Index file not created"

        # Load index
        loaded_store = FAISSVectorStore.load(test_path)
        assert loaded_store.dimension == vector_store.dimension
        assert loaded_store.n_vectors == vector_store.n_vectors
        assert loaded_store.is_trained

        # Test search with loaded index using original vectors
        query = sample_vectors[0:1]  # Use first vector as query
        original_distances, original_indices = vector_store.search(query, k=1)
        loaded_distances, loaded_indices = loaded_store.search(query, k=1)

        # Both should return the same results
        np.testing.assert_array_equal(original_indices, loaded_indices)
        np.testing.assert_array_almost_equal(original_distances, loaded_distances)

    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

def test_index_types(sample_vectors):
    """Test different FAISS index types."""
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    
    # Test each index type
    for index_type in IndexType:
        # Create store with specific index type
        store = FAISSVectorStore(index_type=index_type)
        
        # Add vectors
        store.add(sample_vectors)
        assert store.is_trained
        assert store.index_type == index_type
        
        # Test search
        distances, indices = store.search(query, k=5)
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        
        # Test index-specific parameters
        if index_type == IndexType.IVF_FLAT:
            assert isinstance(store.index, faiss.IndexIVFFlat)
            assert store.index.nprobe == settings.n_probe
            assert store.index.nlist == settings.n_clusters
            
        elif index_type == IndexType.IVF_PQ:
            assert isinstance(store.index, faiss.IndexIVFPQ)
            assert store.index.nprobe == settings.n_probe
            assert store.index.nlist == settings.n_clusters
            assert store.index.pq.M == settings.pq_m
            
        elif index_type == IndexType.HNSW_FLAT:
            assert isinstance(store.index, faiss.IndexHNSWFlat)
            assert store.index.hnsw.efSearch == settings.hnsw_ef_search
            assert store.index.hnsw.efConstruction == settings.hnsw_ef_construction
            
        elif index_type == IndexType.FLAT_L2:
            assert isinstance(store.index, faiss.IndexFlatL2)
            # FlatL2 doesn't have additional parameters to check

def test_search_accuracy():
    """Test search accuracy for different index types."""
    # Generate test data
    n_vectors = 10000
    test_vectors = np.random.randn(n_vectors, settings.vector_dim).astype(np.float32)
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    
    # Get ground truth using FlatL2
    settings.faiss_index_type = IndexType.FLAT_L2
    flat_store = FAISSVectorStore()
    flat_store.add(test_vectors)
    true_distances, true_indices = flat_store.search(query, k=10)
    
    # Test approximate indices
    approximate_indices = [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.HNSW_FLAT]
    for index_type in approximate_indices:
        settings.faiss_index_type = index_type
        store = FAISSVectorStore()
        store.add(test_vectors)
        
        # Search and compare to ground truth
        distances, indices = store.search(query, k=10)
        
        # Calculate recall@10 (how many of the true top-10 we found)
        recall = len(set(indices[0]) & set(true_indices[0])) / 10
        print(f"\nRecall@10 for {index_type}: {recall:.2f}")
        
        # Even approximate indices should have decent recall
        assert recall > 0.3, f"Recall too low for {index_type}"

def test_large_dataset(vector_store, large_vectors):
    """Test handling of large datasets."""
    # Add vectors in batches
    batch_size = 10000
    n_batches = len(large_vectors) // batch_size
    
    start_time = time.time()
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = large_vectors[start_idx:end_idx]
        vector_store.add(batch)
        
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
    # Add vectors in batches
    batch_size = 10000
    for i in range(0, len(large_vectors), batch_size):
        batch = large_vectors[i:i+batch_size]
        vector_store.add(batch)
    
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
        store.add(vectors[:50000])  # Add only half the vectors
        store.save(save_path)  # Save partial index
        
    # Function to simulate crash during add
    def simulate_crash_add(store, vectors):
        store.add(vectors[:10000])  # Add some vectors successfully
        try:
            # Simulate crash by adding vectors with wrong dimension
            bad_vectors = np.random.randn(100, settings.vector_dim + 1).astype(np.float32)
            store.add(bad_vectors)
        except ValueError:
            pass  # Expected error
        return store  # Return store to preserve state
    
    # Test recovery from partial save
    partial_path = str(data_dir / "partial.faiss")
    simulate_crash_save(vector_store, large_vectors, partial_path)
    
    # Load partial index and complete it
    recovered_store = FAISSVectorStore.load(partial_path)
    assert recovered_store.n_vectors == 50000  # Should have half the vectors
    
    # Add remaining vectors
    recovered_store.add(large_vectors[50000:])
    assert recovered_store.n_vectors == len(large_vectors)
    
    # Test recovery from failed add
    crashed_store = simulate_crash_add(FAISSVectorStore(), large_vectors)
    assert crashed_store.n_vectors == 10000  # Should have first 10000 vectors
    
    # Should be able to continue after error
    crashed_store.add(large_vectors[10000:])
    assert crashed_store.n_vectors == len(large_vectors)
    
    # Verify search still works
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    distances, indices = crashed_store.search(query)
    assert len(indices[0]) == settings.n_results
