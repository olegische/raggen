"""Tests for dependency injection in DocumentProcessingService."""
import pytest
import logging

from core.document_processing import DocumentProcessingService
from core.text_splitting.service import TextSplitterService
from core.vector_store.service import VectorStoreService
from tests.di.conftest import MockApplicationContainer, MockRequestContainer

logger = logging.getLogger(__name__)

def test_document_processing_service_creation(app_container, request_container):
    """Test DocumentProcessingService creation through request container."""
    # Get service from request container
    service = request_container.get_document_processing_service()
    
    # Verify service was created with correct dependencies
    assert isinstance(service, DocumentProcessingService), \
        "Should be a DocumentProcessingService"
    assert isinstance(service.text_splitter, TextSplitterService), \
        "Should have TextSplitterService"
    assert isinstance(service.vector_store_service, VectorStoreService), \
        "Should have VectorStoreService"
    assert service.vector_store_service is app_container.get_vector_store_service(), \
        "Should use singleton VectorStoreService from ApplicationContainer"

def test_document_processing_creates_new_instance_per_request(request_container):
    """Test that DocumentProcessingService creates new instance for each request."""
    # Get two instances
    service1 = request_container.get_document_processing_service()
    service2 = request_container.get_document_processing_service()
    
    # Verify they are different instances
    assert service1 is not service2, \
        "Should create new DocumentProcessingService instance for each request"
    
    # And they have different TextSplitterService instances
    assert service1.text_splitter is not service2.text_splitter, \
        "Should create new TextSplitterService for each request"
    
    # But they share the same singleton VectorStoreService
    assert service1.vector_store_service is service2.vector_store_service, \
        "Should use same singleton VectorStoreService"

def test_document_processing_uses_application_vector_store(app_container, request_container):
    """Test that DocumentProcessingService uses singleton VectorStoreService from container."""
    # Get the singleton VectorStoreService
    app_vector_store = app_container.get_vector_store_service()
    
    # Create document processor through request container
    doc_processor = request_container.get_document_processing_service()
    
    # Verify it uses the same VectorStoreService instance
    assert doc_processor.vector_store_service is app_vector_store, \
        "Should use singleton VectorStoreService from ApplicationContainer"

def test_document_processing_requires_dependencies():
    """Test that DocumentProcessingService requires all dependencies."""
    with pytest.raises(ValueError, match="Text splitter must be provided"):
        DocumentProcessingService(text_splitter=None, vector_store_service=None)