"""Request-level dependency container."""
from typing import Optional
from fastapi import Depends

from config.settings import Settings
from core.embeddings import EmbeddingService
from core.text_splitting.service import TextSplitterService
from core.text_splitting.factory import TextSplitStrategyFactory
from core.vector_store.base import VectorStore
from core.vector_store.service import VectorStoreService
from core.document_processing import DocumentProcessingService
from api.documents import (
    DocumentProcessor,
    ProcessingStrategy,
    ParagraphEmbeddingStrategy,
    MergedEmbeddingStrategy,
    CombinedEmbeddingStrategy
)
from .application import ApplicationContainer

class RequestContainer:
    """
    Container for request-level dependencies.
    
    Creates new instances for each request:
    - TextSplitterService: Independent text processing
    - DocumentProcessingService: Independent document processing
    - DocumentProcessor: Strategy-specific processing
    
    Uses singleton services from ApplicationContainer:
    - Settings: Application configuration
    - EmbeddingService: Heavy model and cache
    - VectorStoreService: Shared storage
    """
    
    @staticmethod
    def get_text_splitter_service(
        embedding_service: EmbeddingService = Depends(ApplicationContainer.get_embedding_service),
        settings: Settings = Depends(ApplicationContainer.get_settings)
    ) -> TextSplitterService:
        """
        Create new TextSplitterService for request.
        
        Uses:
        - EmbeddingService singleton for embeddings
        - Settings singleton for configuration
        - New strategy instance for text splitting
        """
        factory = TextSplitStrategyFactory()
        strategy = factory.create(
            settings.text_split_strategy,
            min_length=settings.text_min_length,
            max_length=settings.text_max_length,
            overlap=settings.text_overlap
        )
        return TextSplitterService(
            embedding_service=embedding_service,
            split_strategy=strategy,
            settings=settings
        )
    
    @staticmethod
    def get_document_processing_service(
        text_splitter: TextSplitterService = Depends(get_text_splitter_service),
        vector_store_service: VectorStoreService = Depends(ApplicationContainer.get_vector_store_service)
    ) -> DocumentProcessingService:
        """
        Create new DocumentProcessingService for request.
        
        Uses:
        - New TextSplitterService instance
        - VectorStoreService singleton
        """
        return DocumentProcessingService(
            text_splitter=text_splitter,
            vector_store_service=vector_store_service
        )
    
    @staticmethod
    def get_document_processor(
        strategy: ProcessingStrategy,
        text_splitter: TextSplitterService = Depends(get_text_splitter_service),
        vector_store: Optional[VectorStore] = None
    ) -> DocumentProcessor:
        """
        Create new DocumentProcessor for request.
        
        Args:
            strategy: Processing strategy to use
            text_splitter: New TextSplitterService instance
            vector_store: Optional vector store override
            
        Returns:
            Document processor instance
        """
        if vector_store is None:
            vector_store = ApplicationContainer.get_vector_store_service().store
            
        processors = {
            ProcessingStrategy.PARAGRAPHS: ParagraphEmbeddingStrategy,
            ProcessingStrategy.MERGED: MergedEmbeddingStrategy,
            ProcessingStrategy.COMBINED: CombinedEmbeddingStrategy
        }
        
        processor_class = processors[strategy]
        return processor_class(text_splitter, vector_store)