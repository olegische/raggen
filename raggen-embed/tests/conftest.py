import os
import pytest
import numpy as np
import tempfile
import shutil
import stat
from datetime import datetime
import logging

from src.config.settings import Settings, reset_settings

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
        "FAISS_INDEX_TYPE": "flat_l2"  # По умолчанию точный поиск
    })
    
    # Create settings
    settings = Settings()
    logger.info("[Test] Created settings with FAISS_INDEX_PATH: %s", settings.faiss_index_path)
    
    yield settings
    
    # Cleanup
    logger.info("[Test] Cleaning up temp directory: %s", temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Reset settings and environment
    logger.info("[Test] Resetting settings and environment variables")
    reset_settings()
    for key in [
        "FAISS_INDEX_PATH", "VECTOR_DIM", "N_CLUSTERS", "N_PROBE",
        "PQ_M", "PQ_BITS", "HNSW_M", "HNSW_EF_CONSTRUCTION",
        "HNSW_EF_SEARCH", "N_RESULTS", "FAISS_INDEX_TYPE"
    ]:
        if key in os.environ:
            del os.environ[key]