"""Service for managing vector store lifecycle."""
from typing import Optional, Dict, Any
import logging

from .base import VectorStore
from .factory import VectorStoreFactory
from config.settings import Settings, VectorStoreServiceType

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for managing vector store lifecycle and caching."""
    
    def __init__(
        self,
        settings: Settings,
        factory: VectorStoreFactory,
        base_store: Optional[VectorStore] = None
    ):
        """
        Initialize vector store service.
        
        Args:
            settings: Settings instance
            factory: Vector store factory instance
            base_store: Optional base vector store instance
        """
        self.settings = settings
        self.factory = factory
        self._base_store = base_store
        self._store: Optional[VectorStore] = None
        
    @property
    def store(self) -> VectorStore:
        """
        Get or create vector store instance.
        
        Returns:
            Vector store instance
        """
        if self._store is None:
            logger.info("Creating new vector store instance")
            store_type = self.settings.vector_store_service_type
            self._store = self.factory.create(
                store_type,
                self.settings,
                base_store=self._base_store
            )
            
        return self._store
    
    def reset(self) -> None:
        """Reset the service, clearing any cached instances."""
        logger.info("Resetting vector store service")
        self._store = None