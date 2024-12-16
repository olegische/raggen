from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
import logging
from config.settings import Settings

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self):
        """Initialize vector store."""
        logger.info("[VectorStore] Initializing base vector store")
    
    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the store.
        
        Args:
            vectors: Array of vectors to add
        """
        pass
    
    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors to search for
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the store to disk.
        
        Args:
            path: Path to save to
        """
        logger.info("[VectorStore] Saving vector store to path: %s", path)
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Optional[str] = None, settings: Optional[Settings] = None) -> 'VectorStore':
        """
        Load a store from disk.
        
        Args:
            path: Optional path to load from
            settings: Optional Settings instance
            
        Returns:
            New instance of VectorStore
        """
        logger.info("[VectorStore] Loading vector store from path: %s", path)
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        pass