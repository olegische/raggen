"""Factory for creating text splitting strategies."""
from typing import Dict, Type

from .base import TextSplitStrategy
from .strategies import SlidingWindowStrategy, ParagraphStrategy


class TextSplitStrategyFactory:
    """Factory for creating text splitting strategies."""
    
    _strategies: Dict[str, Type[TextSplitStrategy]] = {
        'sliding_window': SlidingWindowStrategy,
        'paragraph': ParagraphStrategy
    }
    
    @classmethod
    def create(cls, strategy_type: str, **kwargs) -> TextSplitStrategy:
        """
        Create a text splitting strategy.
        
        Args:
            strategy_type: Type of strategy to create ('sliding_window' or 'paragraph')
            **kwargs: Configuration parameters for the strategy
            
        Returns:
            Configured text splitting strategy
            
        Raises:
            ValueError: If strategy type is unknown
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
            
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[TextSplitStrategy]) -> None:
        """
        Register a new strategy type.
        
        Args:
            name: Name for the strategy type
            strategy_class: Class implementing the strategy
        """
        cls._strategies[name] = strategy_class