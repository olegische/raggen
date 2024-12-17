"""Test configuration for dependency injection tests."""
import os
import pytest
import tempfile
import shutil
import stat
import numpy as np

from config.settings import (
    Settings,
    reset_settings,
    VectorStoreServiceType,
    VectorStoreImplementationType
)
from core.vector_store.implementations import FAISSVectorStore
from core.vector_store.service import VectorStoreService
from core.vector_store.factory import VectorStoreFactory

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
def di_settings():
    """Create settings for DI tests."""
    # Reset settings before test
    reset_settings()
    
    # Create temporary directory for FAISS index
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU)
    temp_index_path = os.path.join(temp_dir, "index.faiss")
    
    # Set environment variables
    os.environ.update({
        "FAISS_INDEX_PATH": temp_index_path,
        "VECTOR_DIM": "384",
        "VECTOR_STORE_TYPE": "persistent",
        "VECTOR_STORE_IMPL_TYPE": "faiss"
    })
    
    # Create settings
    settings = Settings()
    settings.vector_store_service_type = VectorStoreServiceType.PERSISTENT
    settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    
    yield settings
    
    # Cleanup
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    reset_settings()
    for key in ["FAISS_INDEX_PATH", "VECTOR_DIM", "VECTOR_STORE_TYPE", "VECTOR_STORE_IMPL_TYPE"]:
        if key in os.environ:
            del os.environ[key]

@pytest.fixture
def app_container(di_settings):
    """Configure and provide ApplicationContainer for DI tests."""
    # Configure container
    MockApplicationContainer.configure(di_settings)
    yield MockApplicationContainer
    # Reset after test
    MockApplicationContainer.reset()

@pytest.fixture
def di_faiss_store(di_settings):
    """Create FAISS store for DI tests."""
    return FAISSVectorStore(settings=di_settings)

@pytest.fixture
def di_vector_store_factory():
    """Create vector store factory for DI tests."""
    return VectorStoreFactory()

@pytest.fixture
def di_vector_store_service(di_settings, di_vector_store_factory, di_faiss_store):
    """Create vector store service with injected FAISS store."""
    return VectorStoreService(
        settings=di_settings,
        factory=di_vector_store_factory,
        base_store=di_faiss_store
    )

@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    n_vectors = 10
    dim = 384
    return np.random.randn(n_vectors, dim).astype(np.float32)