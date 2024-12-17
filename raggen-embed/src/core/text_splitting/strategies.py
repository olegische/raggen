"""Concrete implementations of text splitting strategies."""
from typing import List

from .base import BaseTextSplitStrategy


class SlidingWindowStrategy(BaseTextSplitStrategy):
    """Strategy for splitting text using sliding window with overlap."""
    
    def __init__(self, min_length: int = 100, max_length: int = 1000, overlap: int = 50):
        """
        Initialize sliding window strategy.
        
        Args:
            min_length: Minimum chunk length
            max_length: Maximum chunk length
            overlap: Number of characters to overlap between chunks
        """
        super().__init__(min_length, max_length)
        self.overlap = overlap
    
    def split(self, text: str) -> List[str]:
        """
        Split text using sliding window with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If text is empty or too short
        """
        self._validate_text(text)
        
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


class ParagraphStrategy(BaseTextSplitStrategy):
    """Strategy for splitting text into paragraphs."""
    
    def split(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs
            
        Raises:
            ValueError: If text is empty or paragraphs are too long
        """
        self._validate_text(text)
        
        # Split by double newlines
        paragraphs = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if p]
        
        # Validate paragraph lengths
        if not paragraphs:
            raise ValueError("Text contains no paragraphs")
            
        if max(len(p) for p in paragraphs) > self.max_length:
            raise ValueError("Paragraphs exceed maximum length")
            
        return paragraphs