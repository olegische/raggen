"""Sentence Transformer model implementation."""
import time
import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer
import numpy as np

from ..base import EmbeddingModel
from config.settings import Settings

logger = logging.getLogger(__name__)

class TransformerModel(EmbeddingModel):
    """Sentence Transformer model implementation."""
    
    def __init__(self, model_name: Optional[str] = None, lazy_init: bool = True):
        """
        Initialize transformer model.
        
        Args:
            model_name: Name of the model to use (default: from settings)
            lazy_init: Whether to initialize model lazily
        """
        self._settings = Settings()
        self._model_name = model_name or self._settings.model_name
        self._model: Optional[SentenceTransformer] = None
        self.load_time = 0.0
        
        logger.info("Initializing transformer model: %s", self._model_name)
        
        if not lazy_init:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize model lazily."""
        if self._model is None:
            start_time = time.time()
            self._model = SentenceTransformer(self._model_name)
            self.load_time = time.time() - start_time
            logger.info("Model loaded successfully in %.2f seconds", self.load_time)
    
    @property
    def model(self) -> SentenceTransformer:
        """Get model, initializing if necessary."""
        self._initialize_model()
        return self._model
    
    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            convert_to_numpy: Whether to convert output to numpy array
            
        Returns:
            Array of embeddings
        """
        logger.debug("Encoding %d texts", len(texts))
        start_time = time.time()
        
        embeddings = self.model.encode(texts, convert_to_numpy=convert_to_numpy)
        
        process_time = time.time() - start_time
        logger.debug("Encoding completed in %.2f seconds", process_time)
        
        return embeddings