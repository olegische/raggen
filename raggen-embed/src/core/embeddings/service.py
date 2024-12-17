"""Embedding service implementation."""
import logging
from typing import List, Dict, Optional

import numpy as np

from config.settings import Settings
from .base import EmbeddingService as BaseEmbeddingService, EmbeddingModel, EmbeddingCache
from .implementations import TransformerModel
from .cache import LRUEmbeddingCache

logger = logging.getLogger(__name__)

class EmbeddingService(BaseEmbeddingService):
    """Service for text embeddings generation."""
    
    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        cache: Optional[EmbeddingCache] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize embedding service.
        
        Args:
            model: Embedding model to use (default: TransformerModel)
            cache: Cache implementation to use (default: LRUEmbeddingCache)
            settings: Settings instance (default: new instance)
        """
        self._settings = settings or Settings()
        self._model = model or TransformerModel(lazy_init=True)
        self._cache = cache or LRUEmbeddingCache(
            max_size=self._settings.batch_size * 10
        )
    
    def _validate_input(self, texts: List[str]) -> None:
        """
        Validate input texts.
        
        Args:
            texts: List of texts to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not texts:
            raise ValueError("Empty text list provided")
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text at position {i}")
            if len(text) > self._settings.max_text_length:
                raise ValueError(
                    f"Text at position {i} exceeds maximum length of {self._settings.max_text_length}"
                )
    
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
        try:
            self._validate_input(texts)
            
            logger.debug("Processing batch of %d texts for embeddings", len(texts))
            
            # For batch processing, we don't use cache as it's typically
            # used for different texts each time
            embeddings = self._model.encode(texts)
            
            logger.debug("Generated embeddings shape: %s", str(embeddings.shape))
            
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e), exc_info=True)
            raise
    
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
        try:
            self._validate_input([text])
            
            logger.debug("Processing single text for embedding, length: %d", len(text))
            
            # Try to get from cache
            try:
                return self._cache.get(text)
            except KeyError:
                # Generate new embedding
                embedding = self._model.encode([text])[0]
                self._cache.put(text, embedding)
                return embedding
            
        except Exception as e:
            logger.error("Failed to generate single embedding: %s", str(e), exc_info=True)
            raise
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._cache.get_stats()