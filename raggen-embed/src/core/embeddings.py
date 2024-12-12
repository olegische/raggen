from typing import List, Dict, Optional, Union
import time
import hashlib
from collections import OrderedDict

from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import Settings
from utils.logging import get_logger
from core.text_processing import ParagraphProcessor, ParagraphConfig, Paragraph

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
        
        # Initialize paragraph processor
        self._paragraph_processor = ParagraphProcessor(
            ParagraphConfig(
                max_length=settings.paragraph_max_length,
                min_length=settings.paragraph_min_length,
                overlap=settings.paragraph_overlap,
                preserve_sentences=settings.preserve_sentences
            )
        )
        
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
    
    def get_embeddings(self, texts: List[str], use_paragraphs: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            use_paragraphs: Whether to process texts as paragraphs (default: False)
            
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
            
            if use_paragraphs:
                logger.debug("Using paragraph processing mode")
                all_embeddings = []
                
                for i, text in enumerate(texts):
                    logger.debug("Text %d length: %d characters", i, len(text))
                    
                    # Process text into paragraphs
                    paragraphs = self._paragraph_processor.split_text(text)
                    logger.debug("Text %d split into %d paragraphs", i, len(paragraphs))
                    
                    # Log paragraph details
                    for j, para in enumerate(paragraphs):
                        logger.debug("Text %d, Paragraph %d length: %d characters",
                                   i, j, len(para.text))
                    
                    # Get embeddings for each paragraph
                    paragraph_embeddings = self._get_paragraph_embeddings(paragraphs)
                    logger.debug("Generated embeddings for %d paragraphs in text %d",
                               len(paragraph_embeddings), i)
                    
                    # Merge paragraph embeddings
                    merged_embedding = self._paragraph_processor.merge_embeddings(
                        paragraph_embeddings,
                        strategy=settings.embedding_merge_strategy
                    )
                    logger.debug("Merged embeddings for text %d using strategy: %s",
                               i, settings.embedding_merge_strategy)
                    
                    all_embeddings.append(merged_embedding)
                
                embeddings = np.array(all_embeddings)
            else:
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

    def _get_paragraph_embeddings(self, paragraphs: List[Paragraph]) -> List[List[float]]:
        """
        Generate embeddings for a list of paragraphs.
        
        Args:
            paragraphs: List of Paragraph objects
            
        Returns:
            List of embedding vectors for each paragraph
        """
        # Extract text from paragraphs
        texts = [p.text for p in paragraphs]
        
        # Process in batches to manage memory
        batch_size = settings.batch_size
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug("Processing paragraph batch %d-%d of %d",
                        i, min(i + batch_size, len(texts)), len(texts))
            
            # Log batch text lengths
            for j, text in enumerate(batch_texts):
                logger.debug("Paragraph batch %d, text %d length: %d characters",
                           i // batch_size, j, len(text))
            
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend(batch_embeddings.tolist())
            
            logger.debug("Completed paragraph batch %d-%d",
                        i, min(i + batch_size, len(texts)))
        
        return embeddings
        
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