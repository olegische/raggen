"""Base classes for text splitting strategies."""
from abc import ABC, abstractmethod
from typing import List, Protocol


class TextSplitStrategy(Protocol):
    """Protocol for text splitting strategies."""
    
    def split(self, text: str) -> List[str]:
        """
        Split text according to the strategy.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If text is empty or invalid
        """
        ...


class BaseTextSplitStrategy(ABC):
    """Base class for text splitting strategies."""
    
    def __init__(self, min_length: int = 100, max_length: int = 1000):
        """
        Initialize strategy with configuration.
        
        Args:
            min_length: Minimum chunk length
            max_length: Maximum chunk length
        """
        self.min_length = min_length
        self.max_length = max_length
    
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Split text according to the strategy.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If text is empty or invalid
        """
        pass
    
    def _validate_text(self, text: str) -> None:
        """
        Validate input text.
        
        Args:
            text: Text to validate
            
        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Empty text")