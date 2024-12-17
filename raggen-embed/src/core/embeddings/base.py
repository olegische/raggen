"""Base protocol for embedding services."""
from abc import ABC, abstractmethod
from typing import List, Dict, Protocol
import numpy as np

class EmbeddingCache(Protocol):
    """Protocol for embedding cache implementations."""
    
    def get(self, key: str) -> np.ndarray:
        """
        Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or raises KeyError
        """
        ...
    
    def put(self, key: str, embedding: np.ndarray) -> None:
        """
        Put embedding into cache.
        
        Args:
            key: Cache key
            embedding: Embedding to cache
        """
        ...
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        ...

class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations."""
    
    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            convert_to_numpy: Whether to convert output to numpy array
            
        Returns:
            Array of embeddings
        """
        ...

class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Array of embeddings
            
        Raises:
            ValueError: If input validation fails
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If input validation fails
        """
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        pass