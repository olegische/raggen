"""Tests for FAISS vector store implementation.

This module contains tests for the FAISSVectorStore implementation, including:
- Basic vector store operations (add, search)
- Different FAISS index types (FLAT_L2, IVF_FLAT, IVF_PQ, HNSW_FLAT)
- Performance and accuracy benchmarks
- Vector normalization impact
- Large dataset handling
"""
import os
import signal
import psutil
import time
from contextlib import contextmanager
import numpy as np
import pytest
import faiss

from core.vector_store.base import VectorStore
from core.vector_store.implementations import FAISSVectorStore
from core.vector_store.factory import VectorStoreFactory
from config.settings import Settings, IndexType, VectorStoreType

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_file_size(path):
    """Get file size in MB."""
    return os.path.getsize(path) / 1024 / 1024

def generate_training_vectors(dim: int, n_vectors: int, normalize: bool = True) -> np.ndarray:
    """
    Generate vectors for training with proper scaling.
    
    Args:
        dim: Vector dimension
        n_vectors: Number of vectors to generate
        normalize: Whether to L2-normalize vectors
    """
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    if normalize:
        faiss.normalize_L2(vectors)
    return vectors

@pytest.fixture
def vector_store(test_settings):
    """Fixture for vector store."""
    return FAISSVectorStore(settings=test_settings)

@pytest.fixture
def sample_vectors(test_settings):
    """Fixture for sample vectors."""
    # Generate random vectors for testing
    # FAISS IVF requires at least 39 * n_clusters points for training
    # For IVF_PQ we need at least 39 * 128 = 4992 points
    n_vectors = max(
        39 * test_settings.n_clusters,  # For IVF_FLAT
        39 * 128,  # For IVF_PQ (reduced n_clusters)
        10000  # Minimum reasonable size
    ) + 100  # Add some extra vectors
    return generate_training_vectors(test_settings.vector_dim, n_vectors)

@pytest.fixture
def large_vectors(test_settings):
    """Fixture for large vector dataset."""
    # Generate 100K vectors
    n_vectors = 100_000
    return generate_training_vectors(test_settings.vector_dim, n_vectors)

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

def test_base_interface(vector_store):
    """Test that FAISSVectorStore implements VectorStore interface."""
    assert isinstance(vector_store, VectorStore)
    
    # Check all required methods are implemented
    assert hasattr(vector_store, 'add')
    assert hasattr(vector_store, 'search')
    assert hasattr(vector_store, 'save')
    assert hasattr(vector_store, 'load')
    assert hasattr(vector_store, '__len__')

def test_initialization(test_settings):
    """Test vector store initialization."""
    # Test default initialization
    store = FAISSVectorStore(settings=test_settings)
    assert store.dimension == test_settings.vector_dim
    assert store.n_vectors == 0
    assert not store.is_trained
    
    # Test custom dimension
    custom_dim = 512
    store = FAISSVectorStore(dimension=custom_dim, settings=test_settings)
    assert store.dimension == custom_dim
    assert store.n_vectors == 0
    assert not store.is_trained

def test_adding_vectors(vector_store, sample_vectors, test_settings):
    """Test adding vectors to the index."""
    # Add vectors in batches to test automatic training
    n_add = 10
    vectors_to_add = generate_training_vectors(test_settings.vector_dim, n_add)
    vector_store.add(vectors_to_add)
    assert vector_store.n_vectors == n_add
    assert vector_store.is_trained  # Should be trained after first add
    
    # Add more vectors
    more_vectors = generate_training_vectors(test_settings.vector_dim, 5)
    vector_store.add(more_vectors)
    assert vector_store.n_vectors == n_add + 5
    
    # Test that search works after adding vectors
    query = vectors_to_add[0:1]  # Use first vector as query
    distances, indices = vector_store.search(query, k=1)
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)

def test_searching(vector_store, sample_vectors, test_settings):
    """Test vector similarity search."""
    # Add vectors (should handle training automatically)
    vector_store.add(sample_vectors)
    
    # Test basic search
    query = generate_training_vectors(test_settings.vector_dim, 1)
    distances, indices = vector_store.search(query, k=5)
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    # Test batch search
    queries = generate_training_vectors(test_settings.vector_dim, 3)
    distances, indices = vector_store.search(queries, k=5)
    assert distances.shape == (3, 5)
    assert indices.shape == (3, 5)
    
    # Test search with wrong dimension
    wrong_query = np.random.randn(1, test_settings.vector_dim + 1).astype(np.float32)
    with pytest.raises(ValueError, match="Expected vectors of dimension"):
        vector_store.search(wrong_query)
    
    # Test search without training
    new_store = FAISSVectorStore(settings=test_settings)
    with pytest.raises(RuntimeError, match="Index must be trained"):
        new_store.search(query)
    
    # Test search with k > n_vectors
    k_too_large = len(sample_vectors) + 10
    distances, indices = vector_store.search(query, k=k_too_large)
    assert distances.shape == (1, vector_store.n_vectors)
    assert indices.shape == (1, vector_store.n_vectors)

def test_persistence(vector_store, sample_vectors, test_settings):
    """Test saving and loading the index."""
    # Add vectors (should handle training automatically)
    vector_store.add(sample_vectors)

    # Create test file path in the current directory
    test_path = "test_index.faiss"
    try:
        # Save index
        vector_store.save(test_path)
        assert os.path.exists(test_path), "Index file not created"
        
        # Check index size
        index_size = get_file_size(test_path)
        print(f"\nIndex size: {index_size:.1f} MB")

        # Load index
        loaded_store = FAISSVectorStore.load(test_path, settings=test_settings)
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

@pytest.mark.parametrize("index_type", list(IndexType))
def test_index_types(sample_vectors, test_settings, index_type):
    """Test different FAISS index types."""
    query = generate_training_vectors(test_settings.vector_dim, 1)
    
    # Create store with specific index type
    store = FAISSVectorStore(index_type=index_type, settings=test_settings)
    
    # Add vectors
    store.add(sample_vectors)
    assert store.index_type == index_type
    
    # Test search
    distances, indices = store.search(query, k=5)
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)
    
    # Test index-specific parameters
    if index_type == IndexType.IVF_FLAT:
        assert isinstance(store.index, faiss.IndexIVFFlat)
        assert store.index.nprobe == test_settings.n_probe
        assert store.index.nlist == test_settings.n_clusters
        
    elif index_type == IndexType.IVF_PQ:
        assert isinstance(store.index, faiss.IndexIVFPQ)
        assert store.index.nprobe == test_settings.n_probe
        assert store.index.nlist == test_settings.n_clusters
        assert store.index.pq.M == test_settings.pq_m
        
    elif index_type == IndexType.HNSW_FLAT:
        assert isinstance(store.index, faiss.IndexHNSWFlat)
        assert store.index.hnsw.efSearch == test_settings.hnsw_ef_search
        assert store.index.hnsw.efConstruction == test_settings.hnsw_ef_construction
        
    elif index_type == IndexType.FLAT_L2:
        assert isinstance(store.index, faiss.IndexFlatL2)
        # FlatL2 doesn't have additional parameters to check

@pytest.mark.parametrize("index_type", [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.HNSW_FLAT])
def test_search_accuracy(test_settings, index_type):
    """
    Test search accuracy for different index types.
    
    Каждый тип индекса оптимизирует разные аспекты:
    - IVF_FLAT: Баланс между скоростью и точностью
    - IVF_PQ: Компромисс между памятью и точностью
    - HNSW_FLAT: Оптимизация для высокой точности
    
    Требования к точности учитывают эти особенности:
    - IVF_FLAT: >= 0.4 (хорошая точность при умеренной скорости)
    - IVF_PQ: >= 0.11 (приемлемая точность при существенной экономии памяти)
    - HNSW_FLAT: >= 0.45 (высокая точность для точного поиска)
    """
    # Generate test data
    n_vectors = 20000  # Увеличили размер для лучшего обучения
    
    # Используем нормализацию в зависимости от типа индекса
    normalize = index_type == IndexType.IVF_PQ  # Нормализация только для IVF_PQ
    test_vectors = generate_training_vectors(test_settings.vector_dim, n_vectors, normalize=normalize)
    queries = generate_training_vectors(test_settings.vector_dim, 10, normalize=normalize)
    
    # Get ground truth using FlatL2
    test_settings.faiss_index_type = IndexType.FLAT_L2
    flat_store = FAISSVectorStore(settings=test_settings)
    flat_store.add(test_vectors)
    
    # Warm-up поиск
    _ = flat_store.search(queries[0:1], k=10)
    
    # Получаем ground truth для всех запросов
    true_distances_list = []
    true_indices_list = []
    for i in range(len(queries)):
        distances, indices = flat_store.search(queries[i:i+1], k=10)
        true_distances_list.append(distances)
        true_indices_list.append(indices)
    
    # Замеряем базовое использование памяти
    base_memory = get_memory_usage()
    
    # Настройка параметров для каждого типа индекса
    if index_type == IndexType.IVF_FLAT:
        test_settings.n_clusters = 128  # Уменьшили для лучшего обучения
        test_settings.n_probe = 32      # 25% от кластеров
        min_recall = 0.4               # Баланс точность/скорость
    elif index_type == IndexType.IVF_PQ:
        test_settings.n_clusters = 128  # Уменьшили для лучшего обучения
        test_settings.n_probe = 32      # 25% от кластеров
        test_settings.pq_m = 32         # Меньше сжатие, выше точность
        min_recall = 0.11              # Компромисс память/точность
    elif index_type == IndexType.HNSW_FLAT:
        test_settings.hnsw_m = 32                # Больше соседей
        test_settings.hnsw_ef_construction = 80  # Больше точность при построении
        test_settings.hnsw_ef_search = 128       # Увеличили для лучшей точности
        min_recall = 0.45              # Высокая точность
    
    # Test approximate index
    test_settings.faiss_index_type = index_type
    store = FAISSVectorStore(settings=test_settings)
    
    # Замеряем время добавления
    start_time = time.time()
    store.add(test_vectors)
    add_time = time.time() - start_time
    
    # Замеряем использование памяти
    memory_usage = get_memory_usage() - base_memory
    
    # Сохраняем индекс для измерения размера
    test_path = "test_index.faiss"
    store.save(test_path)
    index_size = get_file_size(test_path)
    os.remove(test_path)
    
    # Warm-up поиск
    _ = store.search(queries[0:1], k=10)
    
    # Замеряем время поиска
    search_times = []
    recalls = []
    for i in range(len(queries)):
        start_time = time.time()
        distances, indices = store.search(queries[i:i+1], k=10)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        recall = len(set(indices[0]) & set(true_indices_list[i][0])) / 10
        recalls.append(recall)
    
    # Считаем метрики
    avg_recall = sum(recalls) / len(recalls)
    avg_search_time = sum(search_times) / len(search_times)
    
    # Выводим результаты
    print(f"\nResults for {index_type}:")
    print(f"Recall@10: {avg_recall:.2f}")
    print(f"Memory usage: {memory_usage:.1f} MB")
    print(f"Index size: {index_size:.1f} MB")
    print(f"Add time: {add_time:.2f}s")
    print(f"Avg search time: {avg_search_time*1000:.2f}ms")
    
    # Проверяем минимальную точность для каждого типа индекса
    assert avg_recall >= min_recall, f"Recall too low for {index_type} (got {avg_recall:.2f}, need >= {min_recall})"
    
    # Проверяем специфичные для каждого типа метрики
    if index_type == IndexType.IVF_PQ:
        # PQ должен давать существенную экономию памяти
        assert index_size < base_memory * 0.25, "PQ index size too large"
        assert avg_search_time < 0.001, "PQ search too slow"
    elif index_type == IndexType.HNSW_FLAT:
        # HNSW должен быть быстрее в поиске чем IVF
        assert avg_search_time < 0.001, "HNSW search too slow"

@pytest.mark.parametrize("normalize", [True, False])
def test_normalization_impact(test_settings, normalize):
    """Test impact of vector normalization on search accuracy."""
    # Generate test data
    n_vectors = 10000
    test_vectors = generate_training_vectors(test_settings.vector_dim, n_vectors, normalize=normalize)
    queries = generate_training_vectors(test_settings.vector_dim, 10, normalize=normalize)
    
    # Get ground truth using FlatL2
    test_settings.faiss_index_type = IndexType.FLAT_L2
    flat_store = FAISSVectorStore(settings=test_settings)
    flat_store.add(test_vectors)
    
    # Test each approximate index
    for index_type in [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.HNSW_FLAT]:
        test_settings.faiss_index_type = index_type
        store = FAISSVectorStore(settings=test_settings)
        store.add(test_vectors)
        
        recalls = []
        for query in queries:
            # Get ground truth
            true_distances, true_indices = flat_store.search(query.reshape(1, -1), k=10)
            
            # Get approximate results
            distances, indices = store.search(query.reshape(1, -1), k=10)
            
            recall = len(set(indices[0]) & set(true_indices[0])) / 10
            recalls.append(recall)
        
        avg_recall = sum(recalls) / len(recalls)
        print(f"\nRecall@10 for {index_type} ({'normalized' if normalize else 'raw'}): {avg_recall:.2f}")

def test_large_dataset(vector_store, large_vectors, test_settings):
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
    query = generate_training_vectors(test_settings.vector_dim, 1)
    with timeout(1):  # Search should complete within 1 second
        distances, indices = vector_store.search(query, k=10)
    assert len(indices[0]) == 10

def test_factory(test_settings):
    """Test vector store factory."""
    
    # Test creating FAISS store
    test_settings.vector_store_type = "faiss"
    store = VectorStoreFactory.create("faiss", test_settings)
    assert isinstance(store, FAISSVectorStore)
    
    # Test creating Persistent store
    test_settings.vector_store_type = "persistent"
    store = VectorStoreFactory.create("persistent", test_settings)
    assert isinstance(store, VectorStore)  # Should be wrapped in PersistentStore
    
    # Test registering new implementation
    class CustomStore(VectorStore):
        def __init__(self, settings): pass
        def add(self, vectors): pass
        def search(self, query_vectors, k=None): pass
        def save(self, path=None): pass
        def load(self, path=None): pass
        def __len__(self): return 0
    
    VectorStoreFactory.register_implementation("custom", CustomStore)
    test_settings.vector_store_type = "custom"
    store = VectorStoreFactory.create("custom", test_settings)
    assert isinstance(store, CustomStore)

def test_factory_unknown_type(test_settings):
    """Test factory with unknown store type."""
    with pytest.raises(ValueError, match="'unknown' is not a valid VectorStoreType"):
        VectorStoreFactory.create("unknown", test_settings)

def test_factory_registration(test_settings):
    """Test factory implementation registration."""
    # Create custom implementation
    class CustomStore(VectorStore):
        def __init__(self, settings): self.settings = settings
        def add(self, vectors): pass
        def search(self, query_vectors, k=None): pass
        def save(self, path=None): pass
        def load(self, path=None): pass
        def __len__(self): return 0
    
    # Register implementation
    VectorStoreFactory.register_implementation("custom", CustomStore)
    
    # Create store using registered implementation
    store = VectorStoreFactory.create("custom", test_settings)
    assert isinstance(store, CustomStore)
    assert store.settings == test_settings
