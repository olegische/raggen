"""Service for text splitting using configurable strategies."""
import logging
import numpy as np
from typing import List, Optional

from .base import TextSplitStrategy
from .factory import TextSplitStrategyFactory
from ..embeddings import EmbeddingService
from ...config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class TextSplitterService:
    """Service for handling text splitting and embeddings."""
    
    def __init__(self,
                 embedding_service: EmbeddingService,
                 split_strategy: Optional[TextSplitStrategy] = None,
                 settings: Optional[Settings] = None):
        """
        Initialize text splitter service.
        
        Args:
            embedding_service: Service for generating embeddings
            split_strategy: Strategy for splitting text (optional)
            settings: Application settings (optional)
        """
        if not embedding_service:
            raise ValueError("Embedding service must be provided")
            
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service
        
        # If no strategy provided, create default from settings
        if split_strategy is None:
            split_strategy = TextSplitStrategyFactory.create(
                str(self.settings.text_split_strategy),
                min_length=self.settings.text_min_length,
                max_length=self.settings.text_max_length,
                overlap=self.settings.text_overlap
            )
            
        self.split_strategy = split_strategy
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using the configured strategy.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If text is empty or invalid
        """
        return self.split_strategy.split(text)
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings for each chunk of text.
        
        Args:
            text: Text to process
            
        Returns:
            Array of embedding vectors for each chunk
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Empty text")
            
        chunks = self.split_text(text)
        if not chunks:
            raise ValueError("Text could not be split into chunks")
            
        embeddings = [self.embedding_service.get_embedding(chunk) for chunk in chunks]
        return np.stack(embeddings)
    
    def merge_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Merge multiple embeddings into one using the configured strategy.
        
        Args:
            embeddings: Array of embedding vectors to merge
            
        Returns:
            Merged embedding vector
            
        Raises:
            ValueError: If no embeddings provided
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
            
        strategy = self.settings.embedding_merge_strategy
        
        if strategy == "mean":
            return np.mean(embeddings, axis=0)
        elif strategy == "weighted":
            # Give 60% weight to first chunk, distribute remaining 40% among others
            first_weight = 0.6
            rest_weight = 1.0 - first_weight
            
            result = first_weight * embeddings[0]
            if len(embeddings) > 1:
                # For remaining chunks, use exponential decay
                rest_weights = np.exp(-np.arange(len(embeddings)-1) * 0.3)
                rest_weights = rest_weights / np.sum(rest_weights)  # Normalize
                weighted_rest = np.average(embeddings[1:], axis=0, weights=rest_weights)
                result += rest_weight * weighted_rest
            
            return result
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")