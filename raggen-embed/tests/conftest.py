import os
import pytest
import numpy as np
import tempfile
import shutil
import stat
from datetime import datetime
import logging
from unittest.mock import Mock

from src.config.settings import (
    Settings,
    reset_settings,
    TextSplitStrategy as StrategyType,
    VectorStoreServiceType,
    VectorStoreImplementationType
)
from src.core.embeddings import EmbeddingService
from src.core.text_splitting import (
    TextSplitStrategy,
    SlidingWindowStrategy,
    ParagraphStrategy,
    TextSplitterService
)
from src.core.vector_store.base import VectorStore

logger = logging.getLogger(__name__)

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

@pytest.fixture(scope="function")
def test_settings():
    """Create settings specifically for tests."""
    logger.info("[Test] Creating test settings")
    
    # Reset settings before test
    reset_settings()
    
    # Create temporary directory with proper permissions
    temp_dir = tempfile.mkdtemp()
    logger.info("[Test] Created temp directory: %s", temp_dir)
    os.chmod(temp_dir, stat.S_IRWXU)  # Read, write, execute for user
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
        # Добавляем настройки для text splitting
        "TEXT_SPLIT_STRATEGY": "sliding_window",
        "TEXT_MIN_LENGTH": "10",
        "TEXT_MAX_LENGTH": "100",
        "TEXT_OVERLAP": "5",
        # Добавляем настройки для vector store
        "VECTOR_STORE_TYPE": "persistent",  # По умолчанию используем persistent store
        "VECTOR_STORE_IMPL_TYPE": "faiss"   # В качестве реализации используем FAISS
    })
    
    # Create settings
    settings = Settings()
    # Устанавливаем типы хранилищ
    settings.vector_store_service_type = VectorStoreServiceType.PERSISTENT
    settings.vector_store_impl_type = VectorStoreImplementationType.FAISS
    logger.info("[Test] Created settings with FAISS_INDEX_PATH: %s", settings.faiss_index_path)
    
    yield settings
    
    # Cleanup
    logger.info("[Test] Cleaning up temp directory: %s", temp_dir)
    if os.path.exists(temp_dir):
        os.chmod(temp_dir, stat.S_IRWXU)  # Восстанавливаем права перед удалением
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    logger.info("[Test] Resetting settings and environment variables")
    reset_settings()
    for key in [
        "FAISS_INDEX_PATH", "VECTOR_DIM", "N_CLUSTERS", "N_PROBE",
        "PQ_M", "PQ_BITS", "HNSW_M", "HNSW_EF_CONSTRUCTION",
        "HNSW_EF_SEARCH", "N_RESULTS", "FAISS_INDEX_TYPE",
        "TEXT_SPLIT_STRATEGY", "TEXT_MIN_LENGTH", "TEXT_MAX_LENGTH",
        "TEXT_OVERLAP", "VECTOR_STORE_TYPE", "VECTOR_STORE_IMPL_TYPE"
    ]:
        if key in os.environ:
            del os.environ[key]

# Text splitting fixtures
@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service.get_embedding.return_value = np.array([1.0, 2.0, 3.0])
    return service

@pytest.fixture
def sliding_window_strategy(test_settings):
    """Create a sliding window strategy using test settings."""
    return SlidingWindowStrategy(
        min_length=test_settings.text_min_length,
        max_length=test_settings.text_max_length,
        overlap=test_settings.text_overlap
    )

@pytest.fixture
def paragraph_strategy(test_settings):
    """Create a paragraph strategy using test settings."""
    return ParagraphStrategy(
        min_length=test_settings.text_min_length,
        max_length=test_settings.text_max_length
    )

@pytest.fixture
def text_splitter_service(mock_embedding_service, sliding_window_strategy, test_settings):
    """Create text splitter service with injected dependencies."""
    return TextSplitterService(
        embedding_service=mock_embedding_service,
        split_strategy=sliding_window_strategy,
        settings=test_settings
    )

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock(spec=VectorStore)
    store.add = Mock()
    store.search = Mock(return_value=(np.array([]), np.array([])))
    store.save = Mock()
    store.load = Mock()
    store.__len__ = Mock(return_value=0)
    return store

@pytest.fixture
def mock_vector_store_factory(mock_vector_store):
    """Create a mock vector store factory."""
    from src.core.vector_store.factory import VectorStoreFactory
    factory = VectorStoreFactory()
    factory._store_implementations[VectorStoreImplementationType.FAISS.value] = lambda _: mock_vector_store
    return factory