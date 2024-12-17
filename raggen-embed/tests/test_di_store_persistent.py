"""Tests for dependency injection in PersistentStore."""
import os
import pytest
import numpy as np

from core.vector_store.implementations import FAISSVectorStore, PersistentStore

def test_persistent_store_with_injected_store(di_settings, di_vector_store_factory, di_faiss_store):
    """Test PersistentStore with injected FAISS store."""
    # Create persistent store with injected base store
    persistent = PersistentStore(
        settings=di_settings,
        factory=di_vector_store_factory,
        store=di_faiss_store
    )
    
    # Verify base store is used
    assert persistent.store is di_faiss_store
    
    # Test operations use the same store
    vectors = np.random.randn(10, di_settings.vector_dim).astype(np.float32)
    persistent.add(vectors)
    
    assert len(di_faiss_store) == 10
    assert len(persistent) == 10

def test_persistent_store_saves_injected_store(di_settings, di_vector_store_factory, di_faiss_store, sample_vectors):
    """Test that PersistentStore correctly saves and loads injected store."""
    persistent = PersistentStore(
        settings=di_settings,
        factory=di_vector_store_factory,
        store=di_faiss_store
    )
    
    # Add vectors
    persistent.add(sample_vectors)
    
    # Save index
    persistent.save()
    assert os.path.exists(di_settings.faiss_index_path)
    
    # Create new persistent store
    new_persistent = PersistentStore(
        settings=di_settings,
        factory=di_vector_store_factory
    )
    
    # Verify loaded store has same data
    assert len(new_persistent) == len(sample_vectors)
    
    # Search should give same results
    query = sample_vectors[0:1]
    original_results = persistent.search(query, k=5)
    loaded_results = new_persistent.search(query, k=5)
    
    np.testing.assert_array_equal(original_results[0], loaded_results[0])  # distances
    np.testing.assert_array_equal(original_results[1], loaded_results[1])  # indices

def test_container_persistent_store_integration(app_container, sample_vectors):
    """Test integration between ApplicationContainer and PersistentStore."""
    # Get components
    base_store = app_container.get_faiss_store()
    service = app_container.get_vector_store_service()
    
    # Add vectors through service
    service.store.add(sample_vectors)
    
    # Verify vectors are in base store
    assert len(base_store) == len(sample_vectors)
    
    # Save through service
    service.store.save()
    assert os.path.exists(app_container.get_settings().faiss_index_path)

def test_persistent_store_auto_creates_store(di_settings, di_vector_store_factory):
    """Test that PersistentStore creates base store if none injected."""
    persistent = PersistentStore(
        settings=di_settings,
        factory=di_vector_store_factory
    )
    
    # Should create FAISS store
    assert isinstance(persistent.store, FAISSVectorStore)
    
    # Should be functional
    vectors = np.random.randn(10, di_settings.vector_dim).astype(np.float32)
    persistent.add(vectors)
    assert len(persistent) == 10

def test_persistent_store_with_existing_index(di_settings, di_vector_store_factory, di_faiss_store, sample_vectors):
    """Test PersistentStore with existing index and injected store."""
    # Create and save initial store
    di_faiss_store.add(sample_vectors)
    di_faiss_store.save(di_settings.faiss_index_path)
    
    # Create persistent store with new base store
    new_base_store = FAISSVectorStore(di_settings)
    persistent = PersistentStore(
        settings=di_settings,
        factory=di_vector_store_factory,
        store=new_base_store
    )
    
    # Should use injected store and load data into it
    assert persistent.store is new_base_store
    assert len(persistent) == len(sample_vectors)