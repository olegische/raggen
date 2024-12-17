"""Tests for FAISS store singleton in ApplicationContainer."""
import pytest
import numpy as np

from config.settings import Settings
from container.application import ApplicationContainer
from core.vector_store.implementations import FAISSVectorStore
from core.vector_store.service import VectorStoreService

@pytest.fixture
def settings():
    """Fixture for test settings."""
    return Settings(
        vector_dim=384,
        faiss_index_path="/tmp/test_index.faiss"
    )

@pytest.fixture
def app_container(settings):
    """Fixture for application container."""
    ApplicationContainer.configure(settings)
    yield ApplicationContainer
    ApplicationContainer.reset()

def test_faiss_store_singleton(app_container):
    """Test that FAISS store is created as singleton."""
    # Get store instance twice
    store1 = app_container.get_faiss_store()
    store2 = app_container.get_faiss_store()
    
    # Should be the same instance
    assert store1 is store2
    assert isinstance(store1, FAISSVectorStore)

def test_faiss_store_in_service(app_container):
    """Test that FAISS store is properly injected into service."""
    # Get store and service
    store = app_container.get_faiss_store()
    service = app_container.get_vector_store_service()
    
    assert isinstance(service, VectorStoreService)
    assert service._base_store is store

def test_faiss_store_persistence(app_container, settings):
    """Test that FAISS store maintains state as singleton."""
    store = app_container.get_faiss_store()
    
    # Add vectors
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    store.add(vectors)
    
    # Get store again and verify vectors are there
    same_store = app_container.get_faiss_store()
    assert len(same_store) == 10
    assert same_store is store

def test_faiss_store_reset(app_container):
    """Test that reset clears FAISS store singleton."""
    # Get initial store
    store1 = app_container.get_faiss_store()
    
    # Reset container
    app_container.reset()
    
    # Get new store
    store2 = app_container.get_faiss_store()
    
    # Should be different instances
    assert store1 is not store2

def test_service_uses_singleton_store(app_container, settings):
    """Test that service uses singleton store for operations."""
    # Get store and service
    store = app_container.get_faiss_store()
    service = app_container.get_vector_store_service()
    
    # Add vectors through service
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    service.store.add(vectors)
    
    # Verify vectors are in singleton store
    assert len(store) == 10
    
    # Search should work through both service and store
    query = np.random.randn(1, settings.vector_dim).astype(np.float32)
    service_results = service.store.search(query, k=5)
    store_results = store.search(query, k=5)
    
    np.testing.assert_array_equal(service_results[0], store_results[0])  # distances
    np.testing.assert_array_equal(service_results[1], store_results[1])  # indices