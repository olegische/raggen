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
        else:
            matches = list(re.finditer(endings, text[:position]))
            if matches:
                return matches[-1].end()
                
        return position

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
        """
        if not text:
            raise ValueError("Empty text provided")
            
        logger.debug("Splitting text of length %d into paragraphs", len(text))
        paragraphs: List[Paragraph] = []
        position = 0
        
        while position < len(text):
            # Determine end position for current paragraph
            end_pos = min(position + self.config.max_length, len(text))
            
            # Adjust boundaries if preserving sentences
            if self.config.preserve_sentences and end_pos < len(text):
                end_pos = self._find_sentence_boundary(text, end_pos)
            
            # Get the paragraph text
            paragraph_text = text[position:end_pos].strip()
            
            # Skip if paragraph is too short (unless it's the last one)
            if len(paragraph_text) < self.config.min_length and end_pos < len(text):
                position += len(paragraph_text)
                continue
            
            # Get context for the paragraph
            context_before, context_after = self._get_context(text, position, end_pos)
            
            # Create paragraph object
            paragraph = Paragraph(
                text=paragraph_text,
                start_pos=position,
                end_pos=end_pos,
                context_before=context_before,
                context_after=context_after
            )
            paragraphs.append(paragraph)
            
            # Move position for next iteration, considering overlap
            position = end_pos - self.config.overlap if end_pos < len(text) else end_pos
            
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
        
        if strategy == "mean":
            return np.mean(paragraph_embeddings, axis=0).tolist()
        elif strategy == "weighted":
            # Simple linear weighting - more weight to earlier paragraphs
            weights = np.linspace(1.0, 0.5, len(paragraph_embeddings))
            weights = weights / np.sum(weights)  # Normalize
            return np.average(paragraph_embeddings, axis=0, weights=weights).tolist()
        else:
            raise ValueError(f"Unknown merging strategy: {strategy}")