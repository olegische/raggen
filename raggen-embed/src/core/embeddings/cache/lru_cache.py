"""LRU cache implementation for embeddings."""
import hashlib
from collections import OrderedDict
from typing import Dict
import numpy as np

from ..base import EmbeddingCache

class LRUEmbeddingCache(EmbeddingCache):
    """LRU (Least Recently Used) cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Create a hash for text to use as cache key.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> np.ndarray:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding
            
        Raises:
            KeyError: If text not in cache
        """
        key = self._hash_text(text)
        try:
            # Move to end to mark as recently used
            embedding = self._cache.pop(key)
            self._cache[key] = embedding
            self._hits += 1
            return embedding
        except KeyError:
            self._misses += 1
            raise
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Put embedding into cache.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding to cache
        """
        key = self._hash_text(text)
        
        # If cache is full, remove oldest item
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            
        self._cache[key] = embedding
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses and total
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
            "size": len(self._cache)
        }