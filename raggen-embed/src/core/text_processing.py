from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from config.settings import Settings
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

@dataclass
class ParagraphConfig:
    """Configuration for paragraph processing."""
    max_length: int = 1000  # Maximum length of a paragraph
    overlap: int = 100      # Number of characters to overlap between paragraphs
    min_length: int = 100   # Minimum length of a paragraph
    preserve_sentences: bool = True  # Whether to preserve sentence boundaries

@dataclass
class Paragraph:
    """Represents a processed paragraph with metadata."""
    text: str
    start_pos: int
    end_pos: int
    context_before: Optional[str] = None
    context_after: Optional[str] = None

class ParagraphProcessor:
    """Handles text splitting into paragraphs with context preservation."""
    
    def __init__(self, config: Optional[ParagraphConfig] = None):
        """Initialize the processor with configuration."""
        self.config = config or ParagraphConfig()
        logger.info("Initializing ParagraphProcessor with config: %s", self.config)
        
    def _find_sentence_boundary(self, text: str, position: int, forward: bool = True) -> int:
        """
        Find the nearest sentence boundary from the given position.
        
        Args:
            text: The text to search in
            position: Starting position
            forward: If True, search forward, otherwise backward
            
        Returns:
            Position of the nearest sentence boundary
        """
        # Common sentence endings
        endings = r'[.!?]+'
        
        if forward:
            matches = list(re.finditer(endings, text[position:]))
            if matches:
                return position + matches[0].end()
            # If no sentence boundary found, return end of text
            return len(text)
        else:
            matches = list(re.finditer(endings, text[:position]))
            if matches:
                return matches[-1].end()
            # If no sentence boundary found going backward, return start of text
            return 0

    def _get_context(self, text: str, start: int, end: int, context_size: int = 200) -> tuple[Optional[str], Optional[str]]:
        """
        Get context before and after the paragraph.
        
        Args:
            text: Full text
            start: Start position of current paragraph
            end: End position of current paragraph
            context_size: Size of context window
            
        Returns:
            Tuple of (context_before, context_after)
        """
        context_before = text[max(0, start - context_size):start].strip() if start > 0 else None
        context_after = text[end:min(len(text), end + context_size)].strip() if end < len(text) else None
        
        return context_before, context_after

    def split_text(self, text: str) -> List[Paragraph]:
        """
        Split text into paragraphs while preserving context.
        
        Args:
            text: Text to split
            
        Returns:
            List of Paragraph objects
            
        Raises:
            ValueError: If text is empty or paragraphs exceed max length
        """
        if not text:
            raise ValueError("Empty text provided")
            
        logger.debug("Splitting text of length %d into paragraphs", len(text))
        paragraphs: List[Paragraph] = []
        
        # Split text into chunks
        start = 0
        while start < len(text):
            # Calculate initial end position
            end = min(start + self.config.max_length, len(text))
            
            # Adjust end position if preserving sentences
            if self.config.preserve_sentences and end < len(text):
                sentence_end = self._find_sentence_boundary(text[start:end], end - start - 1, forward=False)
                if sentence_end > 0:
                    end = start + sentence_end
            
            # Get chunk and validate length
            chunk = text[start:end].strip()
            if len(chunk) > settings.max_text_length:
                # If chunk is too long, try to find a sentence boundary within max_text_length
                if self.config.preserve_sentences:
                    sentence_end = self._find_sentence_boundary(
                        text[start:start + settings.max_text_length],
                        settings.max_text_length - 1,
                        forward=False
                    )
                    if sentence_end > 0:
                        end = start + sentence_end
                        chunk = text[start:end].strip()
                    else:
                        # If no sentence boundary found, force split at max_text_length
                        end = start + settings.max_text_length
                        chunk = text[start:end].strip()
                else:
                    # If not preserving sentences, just cut at max_text_length
                    end = start + settings.max_text_length
                    chunk = text[start:end].strip()
            
            # Skip if chunk is too short (unless it's the last chunk)
            if len(chunk) < self.config.min_length and end < len(text):
                start = end
                continue
            
            # Get context
            context_before = text[max(0, start - 200):start].strip() if start > 0 else None
            context_after = text[end:min(len(text), end + 200)].strip() if end < len(text) else None
            
            # Create paragraph
            paragraph = Paragraph(
                text=chunk,
                start_pos=start,
                end_pos=end,
                context_before=context_before,
                context_after=context_after
            )
            paragraphs.append(paragraph)
            
            # Move to next position, considering overlap
            next_start = end - self.config.overlap if end < len(text) else end
            if next_start <= start:  # Ensure forward progress
                next_start = start + 1
            start = next_start
            
        logger.info("Split text into %d paragraphs", len(paragraphs))
        return paragraphs

    def merge_embeddings(self, paragraph_embeddings: List[List[float]], strategy: str = "mean") -> List[float]:
        """
        Merge embeddings from multiple paragraphs.
        
        Args:
            paragraph_embeddings: List of embedding vectors for each paragraph
            strategy: Merging strategy ("mean" or "weighted")
            
        Returns:
            Merged embedding vector
        """
        if not paragraph_embeddings:
            raise ValueError("No embeddings provided")
            
        import numpy as np
        
        # Convert to numpy array for vectorized operations
        embeddings = np.array(paragraph_embeddings)
        num_paragraphs = len(embeddings)
        
        if strategy == "mean":
            # Simple average of all embeddings
            return np.mean(embeddings, axis=0).tolist()
        elif strategy == "weighted":
            # Create position-based weights with exponential decay
            weights = np.exp(-np.arange(num_paragraphs))
            # Normalize weights
            weights = weights / np.sum(weights)
            # Apply weighted average
            return np.average(embeddings, axis=0, weights=weights).tolist()
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")