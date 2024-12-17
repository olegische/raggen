"""Document processing service."""
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup
import markdown

from core.text_splitting.service import TextSplitterService
from core.vector_store.service import VectorStoreService
from .factory import DocumentProcessorFactory, ProcessingStrategy

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """Service for processing documents."""
    
    def __init__(
        self,
        text_splitter: TextSplitterService,
        vector_store_service: VectorStoreService
    ):
        """
        Initialize document processing service.
        
        Args:
            text_splitter: Service for text splitting and embedding
            vector_store_service: Service for vector storage
        """
        self.text_splitter = text_splitter
        self.vector_store_service = vector_store_service
    
    def process_content(self, content: bytes, file_ext: str) -> str:
        """
        Process file content based on file type.
        
        Args:
            content: Raw file content
            file_ext: File extension (including dot)
            
        Returns:
            Processed text content
            
        Raises:
            ValueError: If content is empty or invalid
        """
        if not content:
            raise ValueError("Empty content")
            
        if file_ext == '.html':
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        elif file_ext == '.md':
            # Convert markdown to HTML first, then extract text
            html = markdown.markdown(content.decode())
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        else:  # .txt
            text = content.decode()
            if not text.strip():
                raise ValueError("Empty text")
            return text
    
    def process_document(
        self,
        text: str,
        strategy: ProcessingStrategy = ProcessingStrategy.PARAGRAPHS
    ) -> Dict[str, Any]:
        """
        Process document text using specified strategy.
        
        Args:
            text: Document text to process
            strategy: Processing strategy to use
            
        Returns:
            Dictionary with processing results
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If processing fails
        """
        if not text.strip():
            raise ValueError("Empty text")
            
        try:
            processor = DocumentProcessorFactory.create(
                strategy,
                self.text_splitter,
                self.vector_store_service.store
            )
            
            logger.info(
                "Processing document with strategy: %s using processor: %s",
                strategy,
                type(processor).__name__
            )
            
            return processor.process(text)
            
        except Exception as e:
            logger.error("Failed to process document: %s", str(e))
            raise RuntimeError(f"Failed to process document: {e}")
    
    def get_supported_types(self) -> Dict[str, Any]:
        """
        Get information about supported document types.
        
        Returns:
            Dictionary with supported types and strategies
        """
        return {
            "supported_types": ['.txt', '.md', '.html'],
            "processing_strategies": [s.value for s in ProcessingStrategy]
        }