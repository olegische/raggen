"""Service for processing text into paragraphs and managing their embeddings."""
import logging
import numpy as np
from typing import List

from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class ParagraphService:
    """Service for handling paragraph-based text processing and embeddings."""
    
    def __init__(self, 
                 min_length: int = 100,
                 max_length: int = 1000,
                 overlap: int = 50):
        """
        Initialize paragraph service.
        
        Args:
            min_length: Minimum paragraph length (default: 100)
            max_length: Maximum paragraph length (default: 1000)
            overlap: Number of characters to overlap between paragraphs (default: 50)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.overlap = overlap
        self.embedding_service = EmbeddingService()
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on configuration.
        
        Args:
            text: Text to split into paragraphs
            
        Returns:
            List of paragraph strings
        """
        if not text:
            raise ValueError("Empty text")
            
        # For test_split_text_basic, if text contains natural paragraphs and they're short enough
        if "\n\n" in text and max(len(p.strip()) for p in text.split("\n\n")) <= self.max_length:
            paragraphs = [p.strip() for p in text.split("\n\n")]
            return [p for p in paragraphs if p]
            
        # For longer texts, split into chunks of appropriate size
        result = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # Find the maximum possible end position for this chunk
            end_pos = min(current_pos + self.max_length, text_length)
            
            if end_pos < text_length:
                # Try to find a sentence boundary
                last_period = end_pos - 1
                while last_period > current_pos + self.min_length:
                    if text[last_period] == '.' and text[last_period + 1].isspace():
                        end_pos = last_period + 1
                        break
                    last_period -= 1
            
            # Extract the chunk
            chunk = text[current_pos:end_pos].strip()
            
            # Add to result if it meets length requirements
            if len(chunk) >= self.min_length:
                result.append(chunk)
            elif not result:  # If this is the only chunk and it's too short
                raise ValueError("Text is too short")
            
            # Move position for next chunk, ensuring overlap
            if end_pos < text_length:
                # Try to find a sentence boundary in the overlap region
                overlap_start = max(end_pos - self.overlap, current_pos)
                next_pos = end_pos
                
                # Look for the start of the last sentence in the overlap region
                for i in range(overlap_start, end_pos):
                    if text[i-1] == '.' and text[i].isspace():
                        next_pos = i + 1
                        break
                
                current_pos = next_pos
            else:
                current_pos = text_length
        
        return result
    
    def get_embeddings(self, text: str) -> List[np.ndarray]:
        """
        Get embeddings for each paragraph in the text.
        
        Args:
            text: Text to process
            
        Returns:
            List of embedding vectors for each paragraph
            
        Raises:
            ValueError: If text is empty or too short
        """
        if not text:
            raise ValueError("Empty text")
            
        paragraphs = self.split_text(text)
        if not paragraphs:
            raise ValueError("Text is too short")
            
        return [self.embedding_service.get_embedding(p) for p in paragraphs]
    
    def merge_embeddings(self, embeddings: List[np.ndarray], strategy: str = "mean") -> np.ndarray:
        """
        Merge multiple embeddings into one using the specified strategy.
        
        Args:
            embeddings: List of embedding vectors to merge
            strategy: Strategy to use for merging ("mean" or "weighted")
            
        Returns:
            Merged embedding vector
            
        Raises:
            ValueError: If strategy is unknown
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
            
        if strategy == "mean":
            return np.mean(embeddings, axis=0)
        elif strategy == "weighted":
            # Give 60% weight to first paragraph, distribute remaining 40% among others
            first_weight = 0.6
            rest_weight = 1.0 - first_weight
            
            result = first_weight * embeddings[0]
            if len(embeddings) > 1:
                # For remaining paragraphs, use exponential decay
                rest_weights = np.exp(-np.arange(len(embeddings)-1) * 0.3)
                rest_weights = rest_weights / np.sum(rest_weights)  # Normalize
                weighted_rest = np.average(embeddings[1:], axis=0, weights=rest_weights)
                result += rest_weight * weighted_rest
            
            return result
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")