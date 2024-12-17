"""Tests for FAISS store singleton in ApplicationContainer."""
import pytest
import numpy as np

from core.vector_store.implementations import FAISSVectorStore
from core.vector_store.service import VectorStoreService
from tests.di.conftest import MockApplicationContainer

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

def test_faiss_store_persistence(app_container, sample_vectors):
    """Test that FAISS store maintains state as singleton."""
    store = app_container.get_faiss_store()
    
    # Add vectors
    store.add(sample_vectors)
    
    # Get store again and verify vectors are there
    same_store = app_container.get_faiss_store()
    assert len(same_store) == len(sample_vectors)
    assert same_store is store
    
    # Search should work
    query = sample_vectors[0:1]
    distances, indices = same_store.search(query, k=1)
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)

def test_faiss_store_reset(di_settings):
    """Test that reset clears FAISS store singleton."""
    # Configure container first time
    MockApplicationContainer.configure(di_settings)
    store1 = MockApplicationContainer.get_faiss_store()
    
    # Reset container
    MockApplicationContainer.reset()
    
    # Configure container second time
    MockApplicationContainer.configure(di_settings)
    store2 = MockApplicationContainer.get_faiss_store()
    
    # Should be different instances
    assert store1 is not store2
    
    # Cleanup
    MockApplicationContainer.reset()

def test_service_uses_singleton_store(app_container, sample_vectors):
    """Test that service uses singleton store for operations."""
    # Get store and service
    store = app_container.get_faiss_store()
    service = app_container.get_vector_store_service()
    
    # Add vectors through base store
    service._base_store.add(sample_vectors)
    
    # Verify vectors are in singleton store
    assert len(store) == len(sample_vectors)
    
    # Search should work through both service and store
    query = sample_vectors[0:1]
    service_results = service._base_store.search(query, k=1)
    store_results = store.search(query, k=1)
    
    np.testing.assert_array_equal(service_results[0], store_results[0])  # distances
    np.testing.assert_array_equal(service_results[1], store_results[1])  # indices