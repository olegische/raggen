"""Test configuration for store tests."""
import os
import pytest
import tempfile
import shutil
import stat
import numpy as np
import logging
import faiss

from src.config.settings import (
    Settings,
    reset_settings,
    VectorStoreServiceType,
    VectorStoreImplementationType
)
from src.core.vector_store.implementations import FAISSVectorStore
from src.core.vector_store.service import VectorStoreService
from src.core.vector_store.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


def generate_training_vectors(dim: int, n_vectors: int, normalize: bool = True) -> np.ndarray:
    """Generate vectors for training with proper scaling."""
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    if normalize:
        faiss.normalize_L2(vectors)
    return vectors


class MockApplicationContainer:
    """Mock container for testing."""
    _settings = None
    _vector_store_service = None
    _vector_store_factory = None
    _faiss_store = None
    
    @classmethod
    def configure(cls, settings):
        cls._settings = settings
        cls._vector_store_factory = VectorStoreFactory()
        cls._faiss_store = FAISSVectorStore(settings)
        cls._vector_store_service = VectorStoreService(
            settings=settings,
            factory=cls._vector_store_factory,
            base_store=cls._faiss_store
        )
    
    @classmethod
    def get_settings(cls):
        return cls._settings
    
    @classmethod
    def get_faiss_store(cls):
        return cls._faiss_store
    
    @classmethod
    def get_vector_store_service(cls):
        return cls._vector_store_service
    
    @classmethod
    def reset(cls):
        cls._settings = None
        cls._vector_store_service = None
        cls._vector_store_factory = None
        cls._faiss_store = None


@pytest.fixture(scope="function")
def store_settings():
    """Create settings for store tests."""
    logger.info("[Test] Creating store test settings")
    
    # Reset settings before test
    reset_settings()
    
    # Create temporary directory for FAISS index
    temp_dir = tempfile.mkdtemp()
    logger.info("[Test] Created temp directory: %s", temp_dir)
    os.chmod(temp_dir, stat.S_IRWXU)
    temp_index_path = os.path.join(temp_dir, "index.faiss")
    logger.info("[Test] Using index path: %s", temp_index_path)
    
    # Set environment variables
    logger.info("[Test] Setting environment variables")
    os.environ.update({
        "FAISS_INDEX_PATH": temp_index_path,
        "VECTOR_DIM": "384",
        "N_CLUSTERS": "128",   # Уменьшили для лучшего обучения
        "N_PROBE": "32",       # 25% от кластеров
        "PQ_M": "32",         # Меньше сжатие, выше точность
        "PQ_BITS": "8",       # Стандартное значение для PQ
        "HNSW_M": "32",       # Больше соседей для точности
        "HNSW_EF_CONSTRUCTION": "80",  # Больше точность при построении
        "HNSW_EF_SEARCH": "128",       # Увеличили для лучшей точности
        "N_RESULTS": "10",
        "FAISS_INDEX_TYPE": "flat_l2",  # По умолчанию точный поиск
        "VECTOR_STORE_TYPE": "persistent",  # По умолчанию используем persistent store
        "VECTOR_STORE_IMPL_TYPE": "faiss"   # В качестве реализации используем FAISS
    })
    
    # Create settings
    settings = Settings()
    settings.vector_store_service_type = VectorStoreServiceType.PERSISTENT
    settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    logger.info("[Test] Created settings with FAISS_INDEX_PATH: %s", settings.faiss_index_path)
    
    yield settings
    
    # Cleanup
    logger.info("[Test] Cleaning up temp directory: %s", temp_dir)
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    logger.info("[Test] Resetting settings and environment variables")
    reset_settings()
    for key in [
        "FAISS_INDEX_PATH", "VECTOR_DIM", "N_CLUSTERS", "N_PROBE",
        "PQ_M", "PQ_BITS", "HNSW_M", "HNSW_EF_CONSTRUCTION",
        "HNSW_EF_SEARCH", "N_RESULTS", "FAISS_INDEX_TYPE",
        "VECTOR_STORE_TYPE", "VECTOR_STORE_IMPL_TYPE"
    ]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def app_container(store_settings):
    """Configure and provide ApplicationContainer for store tests."""
    # Configure container
    MockApplicationContainer.configure(store_settings)
    yield MockApplicationContainer
    # Reset after test
    MockApplicationContainer.reset()


@pytest.fixture
def sample_vectors(store_settings):
    """Fixture for sample vectors."""
    # Generate random vectors for testing
    # FAISS IVF requires at least 39 * n_clusters points for training
    # For IVF_PQ we need at least 39 * 128 = 4992 points
    n_vectors = max(
        39 * store_settings.n_clusters,  # For IVF_FLAT
        39 * 128,  # For IVF_PQ (reduced n_clusters)
        10000  # Minimum reasonable size
    ) + 100  # Add some extra vectors
    return generate_training_vectors(store_settings.vector_dim, n_vectors)


@pytest.fixture
def large_vectors(store_settings):
    """Fixture for large vector dataset."""
    # Generate 100K vectors
    n_vectors = 100_000
    return generate_training_vectors(store_settings.vector_dim, n_vectors)