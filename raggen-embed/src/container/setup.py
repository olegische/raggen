"""FastAPI dependency injection setup."""
import logging
from typing import Callable, Any
from fastapi import FastAPI, Depends

from config.settings import Settings
from core.vector_store.base import VectorStore
from core.vector_store.factory import VectorStoreFactory
from core.text_splitting.service import TextSplitterService
from core.document_processing import DocumentProcessingService
from core.embeddings import EmbeddingService
from .application import ApplicationContainer
from .request import RequestContainer

logger = logging.getLogger(__name__)

def setup_di(app: FastAPI, settings: Settings) -> None:
    """
    Setup dependency injection for FastAPI application.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    logger.info("Configuring dependency injection")
    
    # Configure application container
    ApplicationContainer.configure(settings)
    
    # Register application-level dependencies (синглтоны)
    app.dependency_overrides.update({
        Settings: ApplicationContainer.get_settings,
        VectorStore: lambda: ApplicationContainer.get_vector_store_service().store,
        VectorStoreFactory: ApplicationContainer.get_vector_store_factory,
        EmbeddingService: ApplicationContainer.get_embedding_service,
        VectorStoreService: ApplicationContainer.get_vector_store_service,
    })
    
    # Register request-level dependencies (новый инстанс на каждый запрос)
    app.dependency_overrides.update({
        TextSplitterService: RequestContainer.get_text_splitter_service,
        DocumentProcessingService: RequestContainer.get_document_processing_service,
    })
    
    logger.info("Dependency injection configured")

def cleanup_di() -> None:
    """Cleanup dependency injection resources."""
    logger.info("Cleaning up dependency injection resources")
    ApplicationContainer.reset()
    logger.info("Dependency injection resources cleaned up")