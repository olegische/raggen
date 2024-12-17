"""Factory for creating document processing strategies."""
from enum import Enum
from typing import Dict, Type

from core.text_splitting.service import TextSplitterService
from core.vector_store.base import VectorStore
from .base import DocumentProcessor
from .strategies import (
    ParagraphEmbeddingStrategy,
    MergedEmbeddingStrategy,
    CombinedEmbeddingStrategy
)

class ProcessingStrategy(str, Enum):
    """Available document processing strategies."""
    PARAGRAPHS = "paragraphs"  # Split into paragraphs and save individual embeddings
    MERGED = "merged"          # Split into paragraphs, merge embeddings, save as one
    COMBINED = "combined"      # Do both above strategies

class DocumentProcessorFactory:
    """Factory for creating document processors."""
    
    _implementations: Dict[ProcessingStrategy, Type[DocumentProcessor]] = {
        ProcessingStrategy.PARAGRAPHS: ParagraphEmbeddingStrategy,
        ProcessingStrategy.MERGED: MergedEmbeddingStrategy,
        ProcessingStrategy.COMBINED: CombinedEmbeddingStrategy
    }
    
    @classmethod
    def create(
        cls,
        strategy: ProcessingStrategy,
        text_splitter: TextSplitterService,
        vector_store: VectorStore
    ) -> DocumentProcessor:
        """
        Create document processor instance.
        
        Args:
            strategy: Processing strategy to use
            text_splitter: Text splitter service
            vector_store: Vector store instance
            
        Returns:
            Document processor instance
            
        Raises:
            ValueError: If strategy is unknown
        """
        if not isinstance(strategy, ProcessingStrategy):
            try:
                strategy = ProcessingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Unknown processing strategy: {strategy}")
        
        implementation = cls._implementations[strategy]
        return implementation(text_splitter, vector_store)
    
    @classmethod
    def register_implementation(
        cls,
        strategy_name: str,
        implementation: Type[DocumentProcessor]
    ) -> None:
        """
        Register new processor implementation.
        
        Args:
            strategy_name: Name for the strategy
            implementation: Class implementing DocumentProcessor
        """
        try:
            strategy = ProcessingStrategy(strategy_name)
        except ValueError:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
            
        cls._implementations[strategy] = implementation