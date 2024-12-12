from typing import List, Dict, Optional
import time
import hashlib
from collections import OrderedDict

from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import Settings
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

class EmbeddingService:
    """Service for text embeddings generation."""
    
    def __init__(self, lazy_init: bool = True):
        """
        Initialize the embedding service.
        
        Args:
            lazy_init: Whether to initialize the model lazily (default: True)
        """
        logger.info("Initializing embedding service with model: %s", settings.model_name)
        self._model: Optional[SentenceTransformer] = None
        self._cache_hits = 0
        self._cache_misses = 0
        self.load_time = 0.0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = settings.batch_size * 10
        
        if not lazy_init:
            self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the model lazily."""
        if self._model is None:
            start_time = time.time()
            self._model = SentenceTransformer(settings.model_name)
            self.load_time = time.time() - start_time
            logger.info("Model loaded successfully in %.2f seconds", self.load_time)
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the model, initializing it if necessary."""
        self._initialize_model()
        return self._model
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": self._cache_hits + self._cache_misses
        }
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a hash for text to use as cache key."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _validate_input(self, texts: List[str]) -> None:
        """Validate input texts."""
        if not texts:
            raise ValueError("Empty text list provided")
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text at position {i}")
            if len(text) > settings.max_text_length:
                raise ValueError(f"Text at position {i} exceeds maximum length of {settings.max_text_length}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            numpy.ndarray: Array of embeddings.
            
        Raises:
            ValueError: If input validation fails
            Exception: For other errors during embedding generation
        """
        try:
            self._validate_input(texts)
            
            logger.debug("Processing batch of %d texts for embeddings", len(texts))
            start_time = time.time()
            
            # Log text lengths for debugging
            for i, text in enumerate(texts):
                logger.debug("Text %d length: %d characters", i, len(text))
            
            # For batch processing, we don't use cache as it's typically
            # used for different texts each time
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            process_time = time.time() - start_time
            
            logger.debug("Embeddings generation completed in %.2f seconds", process_time)
            logger.debug("Generated embeddings shape: %s", str(embeddings.shape))
            
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e), exc_info=True)
            raise
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            numpy.ndarray: Embedding vector.
            
        Raises:
            ValueError: If input validation fails
            Exception: For other errors during embedding generation
        """
        try:
            self._validate_input([text])
            
            logger.debug("Processing single text for embedding, length: %d", len(text))
            start_time = time.time()
            
            # Try to get from cache
            text_hash = self._hash_text(text)
            
            if text_hash in self._cache:
                self._cache_hits += 1
                logger.debug("Cache hit for text hash: %s", text_hash)
                # Move to end to mark as recently used
                embedding = self._cache.pop(text_hash)
                self._cache[text_hash] = embedding
            else:
                self._cache_misses += 1
                logger.debug("Cache miss for text hash: %s", text_hash)
                embedding = self.model.encode([text], convert_to_numpy=True)[0]
                
                # Add to cache
                if len(self._cache) >= self._cache_size:
                    # Remove oldest item
                    self._cache.popitem(last=False)
                self._cache[text_hash] = embedding
            
            process_time = time.time() - start_time
            logger.debug("Single embedding generation completed in %.2f seconds", process_time)
            
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate single embedding: %s", str(e), exc_info=True)
            raise