"""Tests for vector store service."""
import pytest
from unittest.mock import Mock

from core.vector_store.service import VectorStoreService
from core.vector_store.base import VectorStore
from core.vector_store.factory import VectorStoreFactory, VectorStoreType
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
    store = Mock(spec=VectorStore)
    return store

@pytest.fixture
def mock_factory(mock_store):
    """Create mock vector store factory."""
    factory = Mock(spec=VectorStoreFactory)
    factory.create.return_value = mock_store
    return factory

def test_service_initialization(test_settings, mock_factory):
    """Test service initialization."""
    service = VectorStoreService(test_settings, mock_factory)
    assert service.settings == test_settings
    assert service.factory == mock_factory
    assert service._store is None

def test_lazy_store_creation(test_settings, mock_factory, mock_store):
    """Test lazy store creation."""
    service = VectorStoreService(test_settings, mock_factory)
    
    # Store should not be created until accessed
    assert service._store is None
    mock_factory.create.assert_not_called()
    
    # Accessing store should create it
    store = service.store
    assert store is mock_store
    mock_factory.create.assert_called_once_with(VectorStoreType(test_settings.vector_store_type), test_settings)
    
    # Second access should return same instance
    assert service.store is store
    mock_factory.create.assert_called_once()

def test_store_reset(test_settings, mock_factory, mock_store):
    """Test store reset."""
    service = VectorStoreService(test_settings, mock_factory)
    
    # Create store
    store1 = service.store
    assert store1 is mock_store
    mock_factory.create.assert_called_once()
    
    # Reset service
    service.reset()
    assert service._store is None
    
    # New store should be created on next access
    store2 = service.store
    assert store2 is mock_store
    assert mock_factory.create.call_count == 2

def test_store_type_change(test_settings, mock_factory, mock_store):
    """Test changing store type."""
    service = VectorStoreService(test_settings, mock_factory)
    
    # Create initial store
    store1 = service.store
    assert store1 is mock_store
    mock_factory.create.assert_called_once_with(VectorStoreType.FAISS, test_settings)
    
    # Change store type
    test_settings.vector_store_type = VectorStoreType.PERSISTENT.value
    service.reset()
    
    # New store should be created with new type
    store2 = service.store
    assert store2 is mock_store
    mock_factory.create.assert_called_with(VectorStoreType.PERSISTENT, test_settings)
    assert mock_factory.create.call_count == 2