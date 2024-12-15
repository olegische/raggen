from abc import ABC, abstractmethod
from typing import Tuple, ClassVar
import numpy as np

class VectorStore(ABC):
    """
    Base interface for vector stores.
    
    The interface defines two types of methods:
    1. Instance methods (add, search, save, __len__) that operate on existing instances
    2. Class methods (load) that create new instances
    
    This separation allows for factory-like creation of stores from saved states
    while maintaining instance-specific operations.
    """
    
    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the store.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
        """
        pass
    
    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of results to return per query
            
        Returns:
            Tuple of (distances, indices)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the store to disk.
        
        This is an instance method as it operates on the current state.
        
        Args:
            path: Path to save the store
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Load a store from disk.
        
        This is a class method as it creates a new instance from saved state.
        
        Args:
            path: Path to load the store from
            
        Returns:
            New instance of the store
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        pass