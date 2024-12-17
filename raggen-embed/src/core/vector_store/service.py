"""Service for managing vector store lifecycle."""
from typing import Optional, Dict, Any
import logging

from .base import VectorStore
from .factory import VectorStoreFactory, VectorStoreType
from config.settings import Settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for managing vector store lifecycle and caching."""
    
    def __init__(self, settings: Settings):
        """
        Initialize vector store service.
        
        Args:
            settings: Settings instance
        """
        self.settings = settings
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
            store_type = VectorStoreType(self.settings.vector_store_type)
            self._store = VectorStoreFactory.create(store_type, self.settings)
            
        return self._store
    
    def reset(self) -> None:
        """Reset the service, clearing any cached instances."""
        logger.info("Resetting vector store service")
        self._store = None