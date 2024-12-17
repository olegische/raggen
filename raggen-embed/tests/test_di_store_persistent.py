"""Tests for dependency injection in PersistentStore."""
import os
import pytest
import numpy as np

from config.settings import Settings, VectorStoreServiceType, VectorStoreImplementationType
from container.application import ApplicationContainer
from core.vector_store.implementations import FAISSVectorStore, PersistentStore
from core.vector_store.factory import VectorStoreFactory

@pytest.fixture
def settings():
    """Fixture for test settings."""
    return Settings(
        vector_dim=384,
        faiss_index_path="/tmp/test_index.faiss",
        vector_store_service_type=VectorStoreServiceType.PERSISTENT,
        vector_store_impl_type=VectorStoreImplementationType.FAISS
    )

@pytest.fixture
def factory():
    """Fixture for vector store factory."""
    return VectorStoreFactory()

@pytest.fixture
def base_store(settings):
    """Fixture for base FAISS store."""
    return FAISSVectorStore(settings)

def test_persistent_store_with_injected_store(settings, factory, base_store):
    """Test PersistentStore with injected FAISS store."""
    # Create persistent store with injected base store
    persistent = PersistentStore(
        settings=settings,
        factory=factory,
        store=base_store
    )
    
    # Verify base store is used
    assert persistent.store is base_store
    
    # Test operations use the same store
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    persistent.add(vectors)
    
    assert len(base_store) == 10
    assert len(persistent) == 10

def test_persistent_store_saves_injected_store(settings, factory, base_store):
    """Test that PersistentStore correctly saves and loads injected store."""
    persistent = PersistentStore(
        settings=settings,
        factory=factory,
        store=base_store
    )
    
    # Add vectors
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    persistent.add(vectors)
    
    try:
        # Save index
        persistent.save()
        assert os.path.exists(settings.faiss_index_path)
        
        # Create new persistent store
        new_persistent = PersistentStore(
            settings=settings,
            factory=factory
        )
        
        # Verify loaded store has same data
        assert len(new_persistent) == 10
        
        # Search should give same results
        query = np.random.randn(1, settings.vector_dim).astype(np.float32)
        original_results = persistent.search(query, k=5)
        loaded_results = new_persistent.search(query, k=5)
        
        np.testing.assert_array_equal(original_results[0], loaded_results[0])  # distances
        np.testing.assert_array_equal(original_results[1], loaded_results[1])  # indices
        
    finally:
        # Cleanup
        if os.path.exists(settings.faiss_index_path):
            os.remove(settings.faiss_index_path)

def test_container_persistent_store_integration(settings):
    """Test integration between ApplicationContainer and PersistentStore."""
    try:
        # Configure container
        ApplicationContainer.configure(settings)
        
        # Get components
        base_store = ApplicationContainer.get_faiss_store()
        service = ApplicationContainer.get_vector_store_service()
        
        # Add vectors through service
        vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
        service.store.add(vectors)
        
        # Verify vectors are in base store
        assert len(base_store) == 10
        
        # Save through service
        service.store.save()
        assert os.path.exists(settings.faiss_index_path)
        
        # Reset container
        ApplicationContainer.reset()
        
        # Configure new container
        ApplicationContainer.configure(settings)
        
        # Get new components
        new_base_store = ApplicationContainer.get_faiss_store()
        new_service = ApplicationContainer.get_vector_store_service()
        
        # Verify data is loaded
        assert len(new_base_store) == 10
        assert len(new_service.store) == 10
        
        # Search should give same results
        query = np.random.randn(1, settings.vector_dim).astype(np.float32)
        original_results = service.store.search(query, k=5)
        new_results = new_service.store.search(query, k=5)
        
        np.testing.assert_array_equal(original_results[0], new_results[0])  # distances
        np.testing.assert_array_equal(original_results[1], new_results[1])  # indices
        
    finally:
        # Cleanup
        ApplicationContainer.reset()
        if os.path.exists(settings.faiss_index_path):
            os.remove(settings.faiss_index_path)

def test_persistent_store_auto_creates_store(settings, factory):
    """Test that PersistentStore creates base store if none injected."""
    persistent = PersistentStore(
        settings=settings,
        factory=factory
    )
    
    # Should create FAISS store
    assert isinstance(persistent.store, FAISSVectorStore)
    
    # Should be functional
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    persistent.add(vectors)
    assert len(persistent) == 10

def test_persistent_store_with_existing_index(settings, factory, base_store):
    """Test PersistentStore with existing index and injected store."""
    # Create and save initial store
    vectors = np.random.randn(10, settings.vector_dim).astype(np.float32)
    base_store.add(vectors)
    base_store.save(settings.faiss_index_path)
    
    try:
        # Create persistent store with new base store
        new_base_store = FAISSVectorStore(settings)
        persistent = PersistentStore(
            settings=settings,
            factory=factory,
            store=new_base_store
        )
        
        # Should use injected store and load data into it
        assert persistent.store is new_base_store
        assert len(persistent) == 10
        
    finally:
        # Cleanup
        if os.path.exists(settings.faiss_index_path):
            os.remove(settings.faiss_index_path)