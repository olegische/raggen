"""Document processing package."""
from .base import DocumentProcessor, EmbeddingMerger, VectorStorer
from .strategies import (
    ParagraphEmbeddingStrategy,
    MergedEmbeddingStrategy,
    CombinedEmbeddingStrategy
)
from .factory import DocumentProcessorFactory, ProcessingStrategy
from .service import DocumentProcessingService

__all__ = [
    # Base
    'DocumentProcessor',
    'EmbeddingMerger',
    'VectorStorer',
    
    # Strategies
    'ParagraphEmbeddingStrategy',
    'MergedEmbeddingStrategy',
    'CombinedEmbeddingStrategy',
    
    # Factory
    'DocumentProcessorFactory',
    'ProcessingStrategy',
    
    # Service
    'DocumentProcessingService'
]