from abc import ABC, abstractmethod
from enum import Enum
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
import markdown
import logging
from typing import List, Dict, Any

from core.paragraph_service import ParagraphService
from core.vector_store.faiss_store import FAISSVectorStore
from core.embeddings import EmbeddingService

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.html'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class ProcessingStrategy(str, Enum):
    """Enum for document processing strategies."""
    PARAGRAPHS = "paragraphs"  # Split into paragraphs and save individual embeddings
    MERGED = "merged"          # Split into paragraphs, merge embeddings, save as one
    COMBINED = "combined"      # Do both above strategies

class DocumentProcessor(ABC):
    """Abstract base class for document processing strategies."""
    
    def __init__(self, paragraph_service: ParagraphService, vector_store: FAISSVectorStore):
        self.paragraph_service = paragraph_service
        self.vector_store = vector_store
    
    @abstractmethod
    def process(self, text: str) -> Dict[str, Any]:
        """Process document text and return results."""
        pass

class ParagraphEmbeddingStrategy(DocumentProcessor):
    """Strategy for processing document into paragraph embeddings."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Split text into paragraphs and get embeddings
        paragraphs = self.paragraph_service.split_text(text)
        embeddings = self.paragraph_service.get_embeddings(text)
        
        # Store embeddings in vector store
        vector_ids = self.vector_store.add_vectors(embeddings)
        
        return {
            "strategy": "paragraphs",
            "paragraphs_count": len(paragraphs),
            "vector_ids": vector_ids,
            "paragraphs": paragraphs
        }

class MergedEmbeddingStrategy(DocumentProcessor):
    """Strategy for processing document into a single merged embedding."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Split text into paragraphs and get embeddings
        paragraphs = self.paragraph_service.split_text(text)
        embeddings = self.paragraph_service.get_embeddings(text)
        
        # Merge embeddings
        merged_embedding = self.paragraph_service.merge_embeddings(embeddings)
        
        # Store merged embedding
        vector_id = self.vector_store.add_vectors([merged_embedding])[0]
        
        return {
            "strategy": "merged",
            "paragraphs_count": len(paragraphs),
            "vector_id": vector_id,
            "paragraphs": paragraphs
        }

class CombinedEmbeddingStrategy(DocumentProcessor):
    """Strategy that combines both paragraph and merged embeddings."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Split text into paragraphs and get embeddings
        paragraphs = self.paragraph_service.split_text(text)
        embeddings = self.paragraph_service.get_embeddings(text)
        
        # Store individual paragraph embeddings
        paragraph_vector_ids = self.vector_store.add_vectors(embeddings)
        
        # Merge embeddings and store
        merged_embedding = self.paragraph_service.merge_embeddings(embeddings)
        merged_vector_id = self.vector_store.add_vectors([merged_embedding])[0]
        
        return {
            "strategy": "combined",
            "paragraphs_count": len(paragraphs),
            "paragraph_vector_ids": paragraph_vector_ids,
            "merged_vector_id": merged_vector_id,
            "paragraphs": paragraphs
        }

def get_paragraph_service() -> ParagraphService:
    """Get or create paragraph service instance."""
    return ParagraphService()

def get_vector_store() -> FAISSVectorStore:
    """Get or create vector store instance."""
    return FAISSVectorStore()

def get_processor(
    strategy: ProcessingStrategy,
    paragraph_service: ParagraphService,
    vector_store: FAISSVectorStore
) -> DocumentProcessor:
    """Factory function to get appropriate processor based on strategy."""
    processors = {
        ProcessingStrategy.PARAGRAPHS: ParagraphEmbeddingStrategy,
        ProcessingStrategy.MERGED: MergedEmbeddingStrategy,
        ProcessingStrategy.COMBINED: CombinedEmbeddingStrategy
    }
    return processors[strategy](paragraph_service, vector_store)

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    strategy: ProcessingStrategy = ProcessingStrategy.PARAGRAPHS,
    paragraph_service: ParagraphService = Depends(get_paragraph_service),
    vector_store: FAISSVectorStore = Depends(get_vector_store)
) -> JSONResponse:
    """
    Upload and process a document file.
    
    The document is processed based on the selected strategy:
    - paragraphs: Split into paragraphs and save individual embeddings
    - merged: Split into paragraphs, merge embeddings, save as one
    - combined: Do both above strategies
    
    Supports: TXT, MD, HTML files
    """
    try:
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )

        # Process content based on file type
        processed_content = ""
        if file_ext == '.html':
            soup = BeautifulSoup(content, 'html.parser')
            processed_content = soup.get_text(separator='\n', strip=True)
        elif file_ext == '.md':
            # Convert markdown to HTML first, then extract text
            html = markdown.markdown(content.decode())
            soup = BeautifulSoup(html, 'html.parser')
            processed_content = soup.get_text(separator='\n', strip=True)
        else:  # .txt
            processed_content = content.decode()

        try:
            # Get appropriate processor and process document
            processor = get_processor(strategy, paragraph_service, vector_store)
            result = processor.process(processed_content)
            
            logger.info(f"Successfully processed document: {file.filename}")
            
            return JSONResponse(
                content={
                    "message": "Document processed successfully",
                    "filename": file.filename,
                    **result
                },
                status_code=200
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing text: {str(e)}"
            )

    except HTTPException as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing document {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/documents/supported-types")
async def get_supported_types() -> JSONResponse:
    """Get list of supported document types"""
    return JSONResponse(
        content={
            "supported_types": list(SUPPORTED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE/1024/1024,
            "processing_strategies": [s.value for s in ProcessingStrategy]
        }
    )
