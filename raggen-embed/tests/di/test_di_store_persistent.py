"""Tests for dependency injection in PersistentStore."""
import os
import pytest
import numpy as np
import logging

from core.vector_store.implementations.faiss import FAISSVectorStore
from core.vector_store.implementations.persistent import PersistentStore
from core.vector_store.service import VectorStoreService
from tests.di.conftest import MockApplicationContainer


logger = logging.getLogger(__name__)

def test_persistent_store_with_injected_store(app_container):
    """Test PersistentStore with injected FAISS store."""
    # Get dependencies from container
    settings = app_container.get_settings()
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create persistent store with injected base store
    store = PersistentStore(
        settings=settings,
        factory=factory,
        store=base_store
    )
    
    # Verify we got a PersistentStore with injected FAISS store
    assert isinstance(store, PersistentStore), "Should be a PersistentStore"
    assert isinstance(store.store, FAISSVectorStore), "Base store should be FAISSVectorStore"
    assert store.store is base_store, "Base store should be the one from container"

def test_persistent_store_saves_injected_store(app_container, sample_vectors):
    """Test that PersistentStore correctly saves and loads injected store."""
    settings = app_container.get_settings()
    logger.info("Test starting with settings:")
    logger.info("  Index path: %s", settings.faiss_index_path)
    logger.info("  Sample vectors shape: %s", sample_vectors.shape)
    
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    logger.info("Initial base store (id: %s):", hex(id(base_store)))
    logger.info("  Type: %s", type(base_store).__name__)
    logger.info("  Length: %d", len(base_store))
    
    # Create persistent store with injected base store
    persistent = PersistentStore(
        settings=settings,
        factory=factory,
        store=base_store,
        auto_save=False  # Disable auto-save to control when we save
    )
    logger.info("Created persistent store (id: %s):", hex(id(persistent)))
    logger.info("  Using base store (id: %s)", hex(id(persistent.store)))
    logger.info("  Initial length: %d", len(persistent))
    
    # Add vectors through persistent store
    persistent.add(sample_vectors)
    logger.info("Added vectors to persistent store (id: %s):", hex(id(persistent)))
    logger.info("  Persistent store length: %d", len(persistent))
    logger.info("  Base store (id: %s) length: %d", hex(id(base_store)), len(base_store))
    
    # Explicitly save
    persistent.save()
    logger.info("Saved persistent store (id: %s)", hex(id(persistent)))
    
    # Verify file exists
    file_exists = os.path.exists(settings.faiss_index_path)
    logger.info("Index file exists: %s", file_exists)
    assert file_exists, "Index file not created"
    
    # Reset container
    app_container.reset()
    logger.info("Reset container")
    
    # Configure new container with same settings
    app_container.configure(settings)
    logger.info("Configured new container with same settings")
    
    # Get new dependencies from container
    new_base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    logger.info("New base store (id: %s):", hex(id(new_base_store)))
    logger.info("  Type: %s", type(new_base_store).__name__)
    logger.info("  Length: %d", len(new_base_store))
    
    # Create new persistent store with new base store
    new_persistent = PersistentStore(
        settings=settings,
        factory=factory,
        store=new_base_store
    )
    logger.info("Created new persistent store (id: %s):", hex(id(new_persistent)))
    logger.info("  Using base store (id: %s)", hex(id(new_persistent.store)))
    logger.info("  Initial length: %d", len(new_persistent))
    
    # Verify data is loaded
    assert len(new_persistent) == len(sample_vectors), "Vectors not loaded from file"
    
    # Search should give same results
    query = sample_vectors[0:1]
    original_results = persistent.search(query, k=5)
    new_results = new_persistent.search(query, k=5)
    
    np.testing.assert_array_equal(original_results[0], new_results[0])  # distances
    np.testing.assert_array_equal(original_results[1], new_results[1])  # indices

def test_persistent_store_auto_creates_store(app_container):
    """Test that PersistentStore creates base store if none injected."""
    settings = app_container.get_settings()
    factory = app_container.get_vector_store_factory()
    
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

def test_persistent_store_with_existing_index(app_container, sample_vectors):
    """Test PersistentStore with existing index."""
    settings = app_container.get_settings()
    logger.info("Index path: %s", settings.faiss_index_path)
    
    # Get dependencies from container
    base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create first persistent store with base store
    persistent = PersistentStore(
        settings=settings,
        factory=factory,
        store=base_store,
        auto_save=False  # Disable auto-save to control when we save
    )
    
    # Add vectors and save explicitly
    persistent.add(sample_vectors)
    persistent.save()
    
    # Verify file exists
    assert os.path.exists(settings.faiss_index_path), "Index file not created"
    
    # Get new dependencies from container
    new_base_store = app_container.get_faiss_store()
    factory = app_container.get_vector_store_factory()
    
    # Create second persistent store with new base store
    new_persistent = PersistentStore(
        settings=settings,
        factory=factory,
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