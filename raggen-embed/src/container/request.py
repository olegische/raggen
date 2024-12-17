"""Request-level dependency container."""
from typing import Optional

from core.text_splitting.service import TextSplitterService
from core.vector_store.base import VectorStore
from api.documents import (
    DocumentProcessor,
    ProcessingStrategy,
    ParagraphEmbeddingStrategy,
    MergedEmbeddingStrategy,
    CombinedEmbeddingStrategy
)
from .application import ApplicationContainer

class RequestContainer:
    """Container for request-level dependencies."""
    
    @staticmethod
    def get_text_splitter_service() -> TextSplitterService:
        """
        Get text splitter service for request.
        
        Returns:
            Text splitter service instance
        """
        settings = ApplicationContainer.get_settings()
        strategy = ApplicationContainer.get_text_split_strategy(
            settings.text_split_strategy
        )
        
        return TextSplitterService(
            embedding_service=ApplicationContainer.get_embedding_service(),
            split_strategy=strategy,
            settings=settings
        )
    
    @staticmethod
    def get_document_processor(
        strategy: ProcessingStrategy,
        text_splitter: TextSplitterService,
        vector_store: Optional[VectorStore] = None
    ) -> DocumentProcessor:
        """
        Get document processor for request.
        
        Args:
            strategy: Processing strategy to use
            text_splitter: Text splitter service
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