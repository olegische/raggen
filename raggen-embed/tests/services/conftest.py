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
    TextSplitStrategy
)
from core.embeddings import DefaultEmbeddingService
from core.embeddings.implementations.transformer_model import TransformerModel
from core.embeddings.cache.lru_cache import LRUEmbeddingCache

logger = logging.getLogger(__name__)

class MockApplicationContainer:
    """Mock container for testing."""
    _settings = None
    _embedding_service = None
    _embedding_model = None
    _embedding_cache = None
    
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
    def reset(cls):
        """Reset container state."""
        cls._settings = None
        cls._embedding_service = None
        cls._embedding_model = None
        cls._embedding_cache = None

@pytest.fixture(scope="function")
def service_settings():
    """Create settings for service tests."""
    logger.info("[Test] Creating service test settings")
    
    # Reset settings before test
    reset_settings()
    
    # Set environment variables
    logger.info("[Test] Setting environment variables")
    os.environ.update({
        # Model settings
        "MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "VECTOR_DIM": "384",
        
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
    logger.info("[Test] Created settings with MODEL_NAME: %s", settings.model_name)
    
    yield settings
    
    # Reset settings and environment
    logger.info("[Test] Resetting settings and environment variables")
    reset_settings()
    for key in [
        # Model settings
        "MODEL_NAME", "VECTOR_DIM",
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
def sample_text():
    """Generate sample text for testing."""
    return """
    This is a sample text that will be used for testing embeddings.
    It needs to be long enough to meet the minimum length requirements
    and contain enough variation to test different aspects of the system.
    Multiple sentences help test proper text handling and embeddings.
    """