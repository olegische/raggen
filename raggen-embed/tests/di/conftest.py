"""Test configuration for dependency injection tests."""
import os
import pytest
import tempfile
import shutil
import stat
import numpy as np
import logging

# Sample data for tests
SAMPLE_TEXT = """
First paragraph with some text.

Second paragraph with different content.

Third paragraph for testing.
"""

SAMPLE_PARAGRAPHS = [
    "First paragraph with some text.",
    "Second paragraph with different content.",
    "Third paragraph for testing."
]

SAMPLE_EMBEDDINGS = np.random.randn(3, 384).astype(np.float32)

logger = logging.getLogger(__name__)

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
from core.text_splitting.service import TextSplitterService
from core.text_splitting.factory import TextSplitStrategyFactory
from core.document_processing.service import DocumentProcessingService

class MockApplicationContainer:
    """
    Mock container for testing application-level dependencies.
    
    Contains only singleton services that should be shared across requests:
    - Settings: Application configuration
    - EmbeddingService: Heavy model initialization and cache
    - VectorStore: Shared vector storage
    - VectorStoreService: Service for vector operations
    - VectorStoreFactory: Factory for creating stores
    """
    
    # Singleton instances
    _settings = None
    _vector_store_service = None
    _vector_store_factory = None
    _faiss_store = None
    _embedding_service = None
    
    @classmethod
    def configure(cls, settings):
        """
        Configure container with application settings.
        
        Initializes only singleton services that are shared across requests:
        - Settings: Application configuration
        - VectorStoreFactory: Factory for creating stores
        - FAISSVectorStore: Base vector store
        - VectorStoreService: Service for vector operations
        - EmbeddingService: Heavy model and cache initialization
        
        Args:
            settings: Application settings
        """
        # Store settings
        cls._settings = settings
        
        # Create vector store dependencies
        cls._vector_store_factory = VectorStoreFactory()
        cls._faiss_store = FAISSVectorStore(settings)
        cls._vector_store_service = VectorStoreService(
            settings=settings,
            factory=cls._vector_store_factory,
            base_store=cls._faiss_store
        )
        
        # Create embedding service with heavy model and cache
        cls._embedding_service = DefaultEmbeddingService(
            settings=settings
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
        """Get embedding service singleton."""
        if cls._embedding_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._embedding_service
    
    @classmethod
    def reset(cls):
        cls._settings = None
        cls._vector_store_service = None
        cls._vector_store_factory = None
        cls._faiss_store = None
        cls._embedding_service = None
        cls._text_splitter_service = None
        cls._document_processing_service = None

class MockRequestContainer:
    """
    Mock request container for testing request-level dependencies.
    
    Creates new instances for each request:
    - TextSplitterService: Independent text processing
    - DocumentProcessingService: Independent document processing
    """
    _app_container = None

    @classmethod
    def configure(cls, app_container):
        """Configure container with application container reference."""
        cls._app_container = app_container

    @classmethod
    def get_text_splitter_service(cls):
        """
        Create new TextSplitterService for request.
        
        Uses:
        - EmbeddingService singleton from ApplicationContainer
        - Settings singleton from ApplicationContainer
        - New strategy instance for text splitting
        """
        if cls._app_container is None:
            raise RuntimeError("Container not configured. Call configure() first.")
            
        settings = cls._app_container.get_settings()
        embedding_service = cls._app_container.get_embedding_service()
        
        factory = TextSplitStrategyFactory()
        strategy = factory.create(
            settings.text_split_strategy,
            min_length=settings.text_min_length,
            max_length=settings.text_max_length,
            overlap=settings.text_overlap
        )
        
        return TextSplitterService(
            embedding_service=embedding_service,
            split_strategy=strategy,
            settings=settings
        )

    @classmethod
    def get_document_processing_service(cls):
        """
        Create new DocumentProcessingService for request.
        
        Uses:
        - New TextSplitterService instance
        - VectorStoreService singleton from ApplicationContainer
        """
        if cls._app_container is None:
            raise RuntimeError("Container not configured. Call configure() first.")
            
        text_splitter = cls.get_text_splitter_service()
        vector_store_service = cls._app_container.get_vector_store_service()
        
        return DocumentProcessingService(
            text_splitter=text_splitter,
            vector_store_service=vector_store_service
        )

    @classmethod
    def reset(cls):
        """Reset container state."""
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