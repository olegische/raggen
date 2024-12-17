"""Embeddings package."""
from .base import EmbeddingService, EmbeddingModel, EmbeddingCache
from .service import EmbeddingService as DefaultEmbeddingService
from .implementations import TransformerModel
from .cache import LRUEmbeddingCache

__all__ = [
    # Base protocols
    'EmbeddingService',
    'EmbeddingModel',
    'EmbeddingCache',
    
    # Default implementations
    'DefaultEmbeddingService',
    'TransformerModel',
    'LRUEmbeddingCache'
]