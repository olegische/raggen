"""Application-level dependency container."""
from typing import Optional, Dict, Type, Union
from core.vector_store.base import VectorStore

from config.settings import Settings
from core.embeddings import EmbeddingService
from core.embeddings.implementations.transformer_model import TransformerModel
from core.embeddings.cache.lru_cache import LRUEmbeddingCache
from core.vector_store.service import VectorStoreService
from core.vector_store.factory import VectorStoreFactory
from core.vector_store.implementations import FAISSVectorStore
from core.text_splitting.factory import TextSplitStrategyFactory
from core.text_splitting.service import TextSplitterService
from core.document_processing import DocumentProcessingService

class ApplicationContainer:
    """Container for application-level dependencies."""
    
    # Singleton instances
    _settings: Optional[Settings] = None
    _vector_store_service: Optional[VectorStoreService] = None
    _vector_store_factory: Optional[VectorStoreFactory] = None
    _faiss_store: Optional[VectorStore] = None
    _embedding_service: Optional[EmbeddingService] = None
    _text_splitter_service: Optional[TextSplitterService] = None
    _document_processing_service: Optional[DocumentProcessingService] = None
    
    @classmethod
    def configure(cls, settings: Settings) -> None:
        """
        Configure container with application settings.
        
        Args:
            settings: Application settings
        """
        cls._settings = settings
        cls._vector_store_factory = VectorStoreFactory()
        
        # Create base FAISS store
        cls._faiss_store = FAISSVectorStore(settings)
        
        # Create vector store service with base store
        cls._vector_store_service = VectorStoreService(
            settings=settings,
            factory=cls._vector_store_factory,
            base_store=cls._faiss_store
        )
        
        # Create embedding service
        model = TransformerModel(lazy_init=True)
        cache = LRUEmbeddingCache(max_size=settings.batch_size * 10)
        cls._embedding_service = EmbeddingService(
            model=model,
            cache=cache,
            settings=settings
        )
        
        # Create text splitter service
        factory = TextSplitStrategyFactory()
        strategy = factory.create(
            settings.text_split_strategy,
            min_length=settings.text_min_length,
            max_length=settings.text_max_length,
            overlap=settings.text_overlap
        )
        cls._text_splitter_service = TextSplitterService(
            embedding_service=cls._embedding_service,
            split_strategy=strategy,
            settings=settings
        )
    
    @classmethod
    def get_vector_store_factory(cls) -> VectorStoreFactory:
        """Get vector store factory."""
        if cls._vector_store_factory is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._vector_store_factory
    
    @classmethod
    def get_settings(cls) -> Settings:
        """Get application settings."""
        if cls._settings is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._settings
    
    @classmethod
    def get_faiss_store(cls) -> VectorStore:
        """Get FAISS vector store singleton."""
        if cls._faiss_store is None:
            if cls._settings is None:
                raise RuntimeError("Container not configured. Call configure() first.")
            from .implementations import FAISSVectorStore
            cls._faiss_store = FAISSVectorStore(cls._settings)
        return cls._faiss_store
    
    @classmethod
    def get_vector_store_service(cls) -> VectorStoreService:
        """Get vector store service."""
        if cls._vector_store_service is None:
            if cls._settings is None:
                raise RuntimeError("Container not configured. Call configure() first.")
            
            # Get or create base FAISS store
            base_store = cls.get_faiss_store()
            
            # Create service with base store
            factory = cls.get_vector_store_factory()
            cls._vector_store_service = VectorStoreService(
                settings=cls._settings,
                factory=factory,
                base_store=base_store
            )
        return cls._vector_store_service
    
    @classmethod
    def get_embedding_service(cls) -> EmbeddingService:
        """Get embedding service."""
        if cls._embedding_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._embedding_service
    
    @classmethod
    def get_text_splitter_service(cls) -> TextSplitterService:
        """Get text splitter service."""
        if cls._text_splitter_service is None:
            raise RuntimeError("Container not configured. Call configure() first.")
        return cls._text_splitter_service
    
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
        cls._vector_store_factory = None
        cls._faiss_store = None
        cls._embedding_service = None
        cls._text_splitter_service = None
        cls._document_processing_service = None