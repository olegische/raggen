"""Application-level dependency container."""
from typing import Optional
from core.vector_store.base import VectorStore

from config.settings import Settings
from core.embeddings import EmbeddingService
from core.embeddings.implementations.transformer_model import TransformerModel
from core.embeddings.cache.lru_cache import LRUEmbeddingCache
from core.vector_store.service import VectorStoreService
from core.vector_store.factory import VectorStoreFactory
from core.vector_store.implementations import FAISSVectorStore

class ApplicationContainer:
    """
    Container for application-level dependencies.
    
    Contains only singleton services that should be shared across requests:
    - Settings: Application configuration
    - EmbeddingService: Heavy model initialization and cache
    - VectorStore: Shared vector storage
    - VectorStoreService: Service for vector operations
    - VectorStoreFactory: Factory for creating stores
    """
    
    # Singleton instances
    _settings: Optional[Settings] = None
    _vector_store_service: Optional[VectorStoreService] = None
    _vector_store_factory: Optional[VectorStoreFactory] = None
    _faiss_store: Optional[VectorStore] = None
    _embedding_service: Optional[EmbeddingService] = None
    
    @classmethod
    def configure(cls, settings: Settings) -> None:
        """
        Configure container with application settings.
        
        Initializes singleton services that are shared across requests:
        - Settings: Application configuration
        - VectorStoreFactory: Factory for creating stores
        - FAISSVectorStore: Base vector store
        - VectorStoreService: Service for vector operations
        - EmbeddingService: Heavy model and cache initialization
        
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
        
        # Create embedding service with heavy model and cache
        model = TransformerModel(lazy_init=True)
        cache = LRUEmbeddingCache(max_size=settings.batch_size * 10)
        cls._embedding_service = EmbeddingService(
            model=model,
            cache=cache,
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
    def reset(cls) -> None:
        """Reset singleton services state."""
        cls._settings = None
        cls._vector_store_service = None
        cls._vector_store_factory = None
        cls._faiss_store = None
        cls._embedding_service = None