import pytest
import os
import tempfile
import shutil
import logging
import numpy as np
from unittest.mock import MagicMock

from src.config.settings import Settings, reset_settings

logger = logging.getLogger(__name__)

# Test data
SAMPLE_TEXT = (
    "First paragraph with some content.\n\n"
    "Second paragraph with different content.\n\n"
    "Third paragraph with more content."
)
SAMPLE_PARAGRAPHS = SAMPLE_TEXT.split("\n\n")
SAMPLE_EMBEDDING = np.ones((384,), dtype=np.float32)
SAMPLE_EMBEDDINGS = np.stack([SAMPLE_EMBEDDING for _ in range(len(SAMPLE_PARAGRAPHS))])

@pytest.fixture(scope="function")
def test_settings():
    """Create settings specifically for tests."""
    # Reset settings before test
    reset_settings()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_index_path = os.path.join(temp_dir, "index.faiss")
    logger.info("[Test] Created temp directory: %s", temp_dir)
    
    # Create settings with temp path
    os.environ["FAISS_INDEX_PATH"] = temp_index_path
    settings = Settings()
    logger.info("[Test] Created settings with FAISS_INDEX_PATH: %s", settings.faiss_index_path)
    
    yield settings
    
    # Cleanup
    logger.info("[Test] Cleaning up temp directory: %s", temp_dir)
    shutil.rmtree(temp_dir)
    
    # Reset settings and environment after test
    reset_settings()
    if "FAISS_INDEX_PATH" in os.environ:
        del os.environ["FAISS_INDEX_PATH"]