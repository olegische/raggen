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
    VectorStoreImplementationType,
    TextSplitStrategy
)
from core.vector_store.implementations.faiss import FAISSVectorStore
from core.vector_store.service import VectorStoreService
from core.vector_store.factory import VectorStoreFactory
from core.embeddings import DefaultEmbeddingService
from core.text_splitting.strategies import SlidingWindowStrategy
from core.text_splitting.service import TextSplitterService
from core.document_processing.service import DocumentProcessingService

class MockApplicationContainer:
    """Mock container for testing."""
    _settings = None
    _vector_store_service = None
    _vector_store_factory = None
    _faiss_store = None
    _embedding_service = None
    _text_split_strategy = None
    _text_splitter_service = None
    _document_processing_service = None
    
    @classmethod
    def configure(cls, settings):
        cls._settings = settings
        
        # Vector store dependencies
        cls._vector_store_factory = VectorStoreFactory()
        cls._faiss_store = FAISSVectorStore(settings)
        cls._vector_store_service = VectorStoreService(
            settings=settings,
            factory=cls._vector_store_factory,
            base_store=cls._faiss_store
        )
        
        # Create embedding service
        cls._embedding_service = DefaultEmbeddingService(
            settings=settings
        )

        # Create text split strategy
        cls._text_split_strategy = SlidingWindowStrategy(
            max_length=settings.text_max_length,
            min_length=settings.text_min_length,
            overlap=settings.text_overlap
        )

        # Create text splitter service
        cls._text_splitter_service = TextSplitterService(
            embedding_service=cls._embedding_service,
            split_strategy=cls._text_split_strategy,
            settings=settings
        )

        # Create document processing service
        cls._document_processing_service = DocumentProcessingService(
            text_splitter=cls._text_splitter_service,
            vector_store_service=cls._vector_store_service
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
    def get_vector_store_factory(cls):
        return cls._vector_store_factory
    
    @classmethod
    def get_embedding_service(cls):
        """Get embedding service."""
        if cls._embedding_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._embedding_service

    @classmethod
    def get_text_split_strategy(cls):
        """Get text split strategy."""
        if cls._text_split_strategy is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._text_split_strategy

    @classmethod
    def get_text_splitter_service(cls):
        """Get text splitter service."""
        if cls._text_splitter_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._text_splitter_service

    @classmethod
    def get_document_processing_service(cls):
        """Get document processing service."""
        if cls._document_processing_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._document_processing_service
    
    @classmethod
    def reset(cls):
        cls._settings = None
        cls._vector_store_service = None
        cls._vector_store_factory = None
        cls._faiss_store = None
        cls._embedding_service = None
        cls._text_split_strategy = None
        cls._text_splitter_service = None
        cls._document_processing_service = None

class MockRequestContainer:
    """Mock request container for testing."""
    _app_container = None

    @classmethod
    def configure(cls, app_container):
        cls._app_container = app_container

    @classmethod
    def get_text_splitter_service(cls):
        """Get text splitter service."""
        if cls._app_container is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._app_container.get_text_splitter_service()

    @classmethod
    def reset(cls):
        cls._app_container = None

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
        # Vector store settings
        "FAISS_INDEX_PATH": temp_index_path,
        "VECTOR_DIM": "384",
        "VECTOR_STORE_TYPE": "persistent",
        "VECTOR_STORE_IMPL_TYPE": "faiss",
        
        # Text splitting settings
        "TEXT_SPLIT_STRATEGY": TextSplitStrategy.SLIDING_WINDOW.value,
        
        # Performance settings
        "TOKENIZERS_PARALLELISM": "false",  # Disable tokenizers parallelism to avoid fork warnings
        "OMP_NUM_THREADS": "1",  # Control OpenMP threads
        "MKL_NUM_THREADS": "1"  # Control MKL threads
    })
    
    # Create settings
    settings = Settings()
    settings.vector_store_service_type = VectorStoreServiceType.PERSISTENT
    settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    settings.text_split_strategy = TextSplitStrategy.SLIDING_WINDOW
    
    yield settings
    
    # Cleanup
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    reset_settings()
    for key in [
        # Vector store settings
        "FAISS_INDEX_PATH", "VECTOR_DIM", "VECTOR_STORE_TYPE", "VECTOR_STORE_IMPL_TYPE",
        # Text splitting settings
        "TEXT_SPLIT_STRATEGY",
        # Performance settings
        "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    ]:
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
def request_container(app_container):
    """Configure and provide RequestContainer for DI tests."""
    # Configure container
    MockRequestContainer.configure(app_container)
    yield MockRequestContainer
    # Reset after test
    MockRequestContainer.reset()

@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    n_vectors = 10
    dim = 384
    return np.random.randn(n_vectors, dim).astype(np.float32)