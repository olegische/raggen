"""Test configuration for service tests."""
import os
import pytest
import tempfile
import shutil
import stat
import numpy as np
import logging

from config.settings import (
    Settings,
    reset_settings,
    TextSplitStrategy,
    VectorStoreServiceType,
    VectorStoreImplementationType
)
from core.embeddings import DefaultEmbeddingService
from core.embeddings.implementations.transformer_model import TransformerModel
from core.embeddings.cache.lru_cache import LRUEmbeddingCache
from core.text_splitting.factory import TextSplitStrategyFactory
from core.text_splitting.service import TextSplitterService
from core.document_processing.service import DocumentProcessingService
from core.vector_store.implementations.faiss import FAISSVectorStore
from core.vector_store.service import VectorStoreService
from core.vector_store.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

class MockApplicationContainer:
    """Mock container for testing."""
    _settings = None
    _embedding_service = None
    _embedding_model = None
    _embedding_cache = None
    _vector_store_service = None
    _vector_store_factory = None
    _faiss_store = None
    
    @classmethod
    def configure(cls, settings):
        """Configure container with settings."""
        cls._settings = settings
        
        # Create embedding service dependencies
        cls._embedding_model = TransformerModel(lazy_init=True)
        cls._embedding_cache = LRUEmbeddingCache(max_size=settings.batch_size * 10)
        
        # Create embedding service
        cls._embedding_service = DefaultEmbeddingService(
            model=cls._embedding_model,
            cache=cls._embedding_cache,
            settings=settings
        )

        # Create vector store dependencies
        cls._vector_store_factory = VectorStoreFactory()
        cls._faiss_store = FAISSVectorStore(settings)
        cls._vector_store_service = VectorStoreService(
            settings=settings,
            factory=cls._vector_store_factory,
            base_store=cls._faiss_store
        )
    
    @classmethod
    def get_settings(cls):
        """Get settings."""
        if cls._settings is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._settings
    
    @classmethod
    def get_embedding_service(cls):
        """Get embedding service."""
        if cls._embedding_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._embedding_service

    @classmethod
    def get_vector_store_service(cls):
        """Get vector store service."""
        if cls._vector_store_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._vector_store_service
    
    @classmethod
    def reset(cls):
        """Reset container state."""
        cls._settings = None
        cls._embedding_service = None
        cls._embedding_model = None
        cls._embedding_cache = None
        cls._vector_store_service = None
        cls._vector_store_factory = None
        cls._faiss_store = None

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
    def get_document_processor(cls, strategy, vector_store=None):
        """
        Create new DocumentProcessor for request.
        
        Args:
            strategy: Processing strategy to use
            vector_store: Optional vector store override
            
        Returns:
            Document processor instance
        """
        if cls._app_container is None:
            raise RuntimeError("Container not configured. Call configure() first.")
            
        text_splitter = cls.get_text_splitter_service()
        if vector_store is None:
            vector_store = cls._app_container.get_vector_store_service().store
            
        from core.document_processing.factory import DocumentProcessorFactory
        return DocumentProcessorFactory.create(strategy, text_splitter, vector_store)

    @classmethod
    def reset(cls):
        """Reset container state."""
        cls._app_container = None

@pytest.fixture(scope="function")
def service_settings():
    """Create settings for service tests."""
    logger.info("[Test] Creating service test settings")
    
    # Reset settings before test
    reset_settings()
    
    # Create temporary directory for FAISS index
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU)
    temp_index_path = os.path.join(temp_dir, "index.faiss")
    
    # Set environment variables
    logger.info("[Test] Setting environment variables")
    os.environ.update({
        # Model settings
        "MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "VECTOR_DIM": "384",
        
        # Vector store settings
        "FAISS_INDEX_PATH": temp_index_path,
        "VECTOR_STORE_TYPE": VectorStoreServiceType.PERSISTENT.value,
        "VECTOR_STORE_IMPL_TYPE": VectorStoreImplementationType.FAISS.value,
        
        # Text splitting settings
        "TEXT_SPLIT_STRATEGY": TextSplitStrategy.SLIDING_WINDOW.value,
        "TEXT_MIN_LENGTH": "100",
        "TEXT_MAX_LENGTH": "1000",
        "TEXT_OVERLAP": "50",
        
        # Performance settings
        "BATCH_SIZE": "32",  # Used to calculate cache size (cache_size = batch_size * 10)
        "MAX_TEXT_LENGTH": "512",  # Maximum length of input text
        "TOKENIZERS_PARALLELISM": "false",  # Disable tokenizers parallelism to avoid fork warnings
        "OMP_NUM_THREADS": "1",  # Control OpenMP threads
        "MKL_NUM_THREADS": "1"  # Control MKL threads
    })
    
    # Create settings
    settings = Settings()
    settings.vector_store_service_type = VectorStoreServiceType.PERSISTENT
    settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    settings.text_split_strategy = TextSplitStrategy.SLIDING_WINDOW
    
    logger.info("[Test] Created settings with MODEL_NAME: %s", settings.model_name)
    
    yield settings
    
    # Cleanup
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    logger.info("[Test] Resetting settings and environment variables")
    reset_settings()
    for key in [
        # Model settings
        "MODEL_NAME", "VECTOR_DIM",
        # Vector store settings
        "FAISS_INDEX_PATH", "VECTOR_STORE_TYPE", "VECTOR_STORE_IMPL_TYPE",
        # Text splitting settings
        "TEXT_SPLIT_STRATEGY", "TEXT_MIN_LENGTH", "TEXT_MAX_LENGTH", "TEXT_OVERLAP",
        # Performance settings
        "BATCH_SIZE", "MAX_TEXT_LENGTH", "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    ]:
        if key in os.environ:
            del os.environ[key]

@pytest.fixture
def app_container(service_settings):
    """Configure and provide ApplicationContainer for service tests."""
    # Configure container
    MockApplicationContainer.configure(service_settings)
    yield MockApplicationContainer
    # Reset after test
    MockApplicationContainer.reset()

@pytest.fixture
def request_container(app_container):
    """Configure and provide RequestContainer for service tests."""
    # Configure container
    MockRequestContainer.configure(app_container)
    yield MockRequestContainer
    # Reset after test
    MockRequestContainer.reset()

@pytest.fixture
def sample_text():
    """Generate sample text for testing."""
    return """
    This is a sample text that will be used for testing embeddings.
    It needs to be long enough to meet the minimum length requirements
    and contain enough variation to test different aspects of the system.
    Multiple sentences help test proper text handling and embeddings.
    """