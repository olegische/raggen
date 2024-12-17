"""Application-level dependency container."""
from typing import Optional, Dict, Type

from config.settings import Settings
from core.embeddings import EmbeddingService
from core.vector_store.service import VectorStoreService
from core.text_splitting.factory import TextSplitStrategyFactory
from core.text_splitting.base import TextSplitStrategy
from core.document_processing import DocumentProcessingService

class ApplicationContainer:
    """Container for application-level dependencies."""
    
    # Singleton instances
    _settings: Optional[Settings] = None
    _vector_store_service: Optional[VectorStoreService] = None
    _embedding_service: Optional[EmbeddingService] = None
    _text_split_factory: Optional[TextSplitStrategyFactory] = None
    _document_processing_service: Optional[DocumentProcessingService] = None
    
    # Strategy cache
    _text_split_strategies: Dict[str, TextSplitStrategy] = {}
    
    @classmethod
    def configure(cls, settings: Settings) -> None:
        """
        Configure container with application settings.
        
        Args:
            settings: Application settings
        """
        cls._settings = settings
        cls._vector_store_service = VectorStoreService(settings)
        cls._embedding_service = EmbeddingService()
        cls._text_split_factory = TextSplitStrategyFactory()
    
    @classmethod
    def get_settings(cls) -> Settings:
        """Get application settings."""
        if cls._settings is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._settings
    
    @classmethod
    def get_vector_store_service(cls) -> VectorStoreService:
        """Get vector store service."""
        if cls._vector_store_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._vector_store_service
    
    @classmethod
    def get_embedding_service(cls) -> EmbeddingService:
        """Get embedding service."""
        if cls._embedding_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._embedding_service
    
    @classmethod
    def get_text_split_strategy(cls, strategy_type: str) -> TextSplitStrategy:
        """
        Get text split strategy (cached).
        
        Args:
            strategy_type: Type of strategy to get
            
        Returns:
            Text split strategy instance
        """
        if cls._text_split_factory is None:
            raise RuntimeError("Container not configured. Call configure() first.")
            
        if strategy_type not in cls._text_split_strategies:
            cls._text_split_strategies[strategy_type] = cls._text_split_factory.create(
                strategy_type,
                min_length=cls._settings.text_min_length,
                max_length=cls._settings.text_max_length,
                overlap=cls._settings.text_overlap
            )
        
        return cls._text_split_strategies[strategy_type]
    
    @classmethod
    def get_document_processing_service(cls) -> DocumentProcessingService:
        """Get document processing service."""
        if cls._document_processing_service is None:
            text_splitter = cls.get_text_splitter_service()
            vector_store_service = cls.get_vector_store_service()
            cls._document_processing_service = DocumentProcessingService(
                text_splitter,
                vector_store_service
            )
        return cls._document_processing_service
    
    @classmethod
    def reset(cls) -> None:
        """Reset container state."""
        cls._settings = None
        cls._vector_store_service = None
        cls._embedding_service = None
        cls._text_split_factory = None
        cls._document_processing_service = None
        cls._text_split_strategies.clear()