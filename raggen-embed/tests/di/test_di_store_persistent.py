"""Tests for dependency injection in PersistentStore."""
import os
import pytest
import numpy as np
import logging

from core.vector_store.implementations import FAISSVectorStore, PersistentStore
from core.vector_store.service import VectorStoreService
from tests.di.conftest import MockApplicationContainer

logger = logging.getLogger(__name__)

def test_persistent_store_with_injected_store(app_container):
    """Test PersistentStore with injected FAISS store."""
    # Get store and service
    base_store = app_container.get_faiss_store()
    service = app_container.get_vector_store_service()
    
    # Verify base store is injected
    assert isinstance(service._base_store, FAISSVectorStore)
    assert service._base_store is base_store

def test_persistent_store_saves_injected_store(app_container, di_vector_store_factory, sample_vectors):
    """Test that PersistentStore correctly saves and loads injected store."""
    settings = app_container.get_settings()
    logger.info("Index path: %s", settings.faiss_index_path)
    
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create persistent store with injected base store
    persistent = PersistentStore(
        settings=settings,
        factory=di_vector_store_factory,
        store=base_store,
        auto_save=False  # Disable auto-save to control when we save
    )
    
    # Add vectors through persistent store
    persistent.add(sample_vectors)
    
    # Explicitly save
    persistent.save()
    
    # Verify file exists
    assert os.path.exists(settings.faiss_index_path), "Index file not created"
    
    # Reset container
    MockApplicationContainer.reset()
    
    # Configure new container with same settings
    MockApplicationContainer.configure(settings)
    
    # Get new base store from container
    new_base_store = app_container.get_faiss_store()
    
    # Create new persistent store with new base store
    new_persistent = PersistentStore(
        settings=settings,
        factory=di_vector_store_factory,
        store=new_base_store
    )
    
    # Verify data is loaded
    assert len(new_persistent) == len(sample_vectors), "Vectors not loaded from file"
    
    # Search should give same results
    query = sample_vectors[0:1]
    original_results = persistent.search(query, k=5)
    new_results = new_persistent.search(query, k=5)
    
    np.testing.assert_array_equal(original_results[0], new_results[0])  # distances
    np.testing.assert_array_equal(original_results[1], new_results[1])  # indices

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

def test_persistent_store_with_existing_index(app_container, di_vector_store_factory, sample_vectors):
    """Test PersistentStore with existing index."""
    settings = app_container.get_settings()
    logger.info("Index path: %s", settings.faiss_index_path)
    
    # Get base store from container
    base_store = app_container.get_faiss_store()
    
    # Create first persistent store with base store
    persistent = PersistentStore(
        settings=settings,
        factory=di_vector_store_factory,
        store=base_store,
        auto_save=False  # Disable auto-save to control when we save
    )
    
    # Add vectors and save explicitly
    persistent.add(sample_vectors)
    persistent.save()
    
    # Verify file exists
    assert os.path.exists(settings.faiss_index_path), "Index file not created"
    
    # Get new base store from container
    new_base_store = app_container.get_faiss_store()
    
    # Create second persistent store with new base store
    new_persistent = PersistentStore(
        settings=settings,
        factory=di_vector_store_factory,
        store=new_base_store
    )
    
    # Should load data automatically
    assert len(new_persistent) == len(sample_vectors), "Vectors not loaded from file"
    
    # Search should give same results
    query = sample_vectors[0:1]
    original_results = persistent.search(query, k=5)
    new_results = new_persistent.search(query, k=5)
    
    np.testing.assert_array_equal(original_results[0], new_results[0])  # distances
    np.testing.assert_array_equal(original_results[1], new_results[1])  # indices