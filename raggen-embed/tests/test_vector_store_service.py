"""Tests for vector store service."""
import pytest
from unittest.mock import MagicMock

from core.vector_store.service import VectorStoreService
from core.vector_store.base import VectorStore
from core.vector_store.factory import VectorStoreType
from config.settings import Settings

@pytest.fixture
def test_settings():
    """Create test settings."""
    settings = Settings()
    settings.vector_store_type = VectorStoreType.FAISS.value
    return settings

@pytest.fixture
def mock_store():
    """Create mock vector store."""
    store = MagicMock(spec=VectorStore)
    return store

def test_service_initialization(test_settings):
    """Test service initialization."""
    service = VectorStoreService(test_settings)
    assert service.settings == test_settings
    assert service._store is None

def test_lazy_store_creation(test_settings):
    """Test lazy store creation."""
    service = VectorStoreService(test_settings)
    
    # Store should not be created until accessed
    assert service._store is None
    
    # Accessing store should create it
    store = service.store
    assert store is not None
    assert isinstance(store, VectorStore)
    
    # Second access should return same instance
    assert service.store is store

def test_store_reset(test_settings):
    """Test store reset."""
    service = VectorStoreService(test_settings)
    
    # Create store
    store1 = service.store
    assert store1 is not None
    
    # Reset service
    service.reset()
    assert service._store is None
    
    # New store should be different instance
    store2 = service.store
    assert store2 is not None
    assert store2 is not store1

def test_store_type_change(test_settings):
    """Test changing store type."""
    service = VectorStoreService(test_settings)
    
    # Create initial store
    store1 = service.store
    assert store1 is not None
    
    # Change store type
    test_settings.vector_store_type = VectorStoreType.PERSISTENT.value
    service.reset()
    
    # New store should be different type
    store2 = service.store
    assert store2 is not None
    assert store2 is not store1
    assert type(store2) != type(store1)