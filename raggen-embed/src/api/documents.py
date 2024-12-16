from abc import ABC, abstractmethod
from enum import Enum
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
import markdown
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from core.paragraph_service import ParagraphService
from core.vector_store.persistent_store import PersistentStore
from core.vector_store.base import VectorStore
from core.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreType
from core.embeddings import EmbeddingService
from config.settings import Settings, get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

# File size limits
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

# Supported file types
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.html'}

class ProcessingStrategy(str, Enum):
    """Enum for document processing strategies."""
    PARAGRAPHS = "paragraphs"  # Split into paragraphs and save individual embeddings
    MERGED = "merged"          # Split into paragraphs, merge embeddings, save as one
    COMBINED = "combined"      # Do both above strategies

class TextProcessor:
    """Base class for text processing operations."""
    
    def __init__(self, paragraph_service: ParagraphService):
        self.paragraph_service = paragraph_service
    
    def process_text(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Process text into paragraphs and embeddings."""
        paragraphs = self.paragraph_service.split_text(text)
        embeddings = self.paragraph_service.get_embeddings(text)
        return paragraphs, embeddings

class EmbeddingMerger:
    """Mixin for merging embeddings."""
    
    def merge_embeddings(self, paragraph_service: ParagraphService, embeddings: np.ndarray) -> np.ndarray:
        """Merge multiple embeddings into one."""
        return paragraph_service.merge_embeddings(embeddings)

class VectorStorer:
    """Mixin for vector storage operations."""
    
    @staticmethod
    def store_vectors(vector_store: PersistentStore, vectors: np.ndarray) -> List[int]:
        """Store vectors and return their IDs."""
        try:
            logger.info("[VectorStorer] Storing vectors with shape: %s using store type: %s", 
                       vectors.shape, type(vector_store).__name__)
            vector_store.add(vectors)
            return list(range(len(vectors)))  # Simplified ID generation
        except Exception as e:
            logger.error("[VectorStorer] Failed to store vectors: %s", str(e))
            raise RuntimeError(f"Failed to store vectors: {e}")
    
    @staticmethod
    def store_single_vector(vector_store: PersistentStore, vector: np.ndarray) -> int:
        """Store a single vector and return its ID."""
        try:
            logger.info("[VectorStorer] Storing single vector with shape: %s using store type: %s", 
                       vector.shape, type(vector_store).__name__)
            vector_store.add(vector.reshape(1, -1))
            return 0  # Simplified ID generation
        except Exception as e:
            logger.error("[VectorStorer] Failed to store single vector: %s", str(e))
            raise RuntimeError(f"Failed to store single vector: {e}")

class DocumentProcessor(ABC, TextProcessor):
    """Abstract base class for document processing strategies."""
    
    def __init__(self, paragraph_service: ParagraphService, vector_store: PersistentStore):
        super().__init__(paragraph_service)
        self.vector_store = vector_store
        logger.info("[DocumentProcessor] Created with store type: %s", type(vector_store).__name__)
    
    @abstractmethod
    def process(self, text: str) -> Dict[str, Any]:
        """Process document text and return results."""
        pass

class ParagraphEmbeddingStrategy(DocumentProcessor, VectorStorer):
    """Strategy for processing document into paragraph embeddings."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Process text into paragraphs and embeddings
        paragraphs, embeddings = self.process_text(text)
        
        # Store embeddings in vector store
        vector_ids = self.store_vectors(self.vector_store, embeddings)
        
        return {
            "strategy": "paragraphs",
            "paragraphs_count": len(paragraphs),
            "vector_ids": vector_ids,
            "paragraphs": paragraphs
        }

class MergedEmbeddingStrategy(DocumentProcessor, EmbeddingMerger, VectorStorer):
    """Strategy for processing document into a single merged embedding."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Process text into paragraphs and embeddings
        paragraphs, embeddings = self.process_text(text)
        
        # Merge embeddings and store
        merged_embedding = self.merge_embeddings(self.paragraph_service, embeddings)
        vector_id = self.store_single_vector(self.vector_store, merged_embedding)
        
        return {
            "strategy": "merged",
            "paragraphs_count": len(paragraphs),
            "vector_id": vector_id,
            "paragraphs": paragraphs
        }

class CombinedEmbeddingStrategy(DocumentProcessor, EmbeddingMerger, VectorStorer):
    """Strategy that combines both paragraph and merged embeddings."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Process text into paragraphs and embeddings
        paragraphs, embeddings = self.process_text(text)
        
        # Store individual paragraph embeddings
        paragraph_vector_ids = self.store_vectors(self.vector_store, embeddings)
        
        # Merge embeddings and store
        merged_embedding = self.merge_embeddings(self.paragraph_service, embeddings)
        merged_vector_id = self.store_single_vector(self.vector_store, merged_embedding)
        
        return {
            "strategy": "combined",
            "paragraphs_count": len(paragraphs),
            "paragraph_vector_ids": paragraph_vector_ids,
            "merged_vector_id": merged_vector_id,
            "paragraphs": paragraphs
        }

def get_paragraph_service(settings: Settings = Depends(get_settings)) -> ParagraphService:
    """Get paragraph service instance."""
    logger.info("[Documents API] Creating paragraph service")
    embedding_service = EmbeddingService()
    return ParagraphService(embedding_service=embedding_service)

def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStore:
    """Get vector store instance."""
    logger.info("[Documents API] Creating vector store with settings: %s", settings.faiss_index_path)
    return VectorStoreFactory.create(VectorStoreType.FAISS, settings)

def get_persistent_store(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store)
) -> PersistentStore:
    """Get persistent store instance."""
    logger.info("[Documents API] Creating persistent store")
    logger.info("[Documents API] Using FAISS_INDEX_PATH from settings: %s", settings.faiss_index_path)
    logger.info("[Documents API] Using vector_store type: %s", type(vector_store).__name__)
    
    store = VectorStoreFactory.create(VectorStoreType.PERSISTENT, settings)
    logger.info("[Documents API] Persistent store created with index_path: %s", store.index_path)
    return store

def get_processor(
    strategy: ProcessingStrategy,
    paragraph_service: ParagraphService,
    vector_store: PersistentStore
) -> DocumentProcessor:
    """Factory function to get appropriate processor based on strategy."""
    logger.info("[Documents API] Creating processor for strategy: %s with vector_store type: %s", 
                strategy, type(vector_store).__name__)
    processors = {
        ProcessingStrategy.PARAGRAPHS: ParagraphEmbeddingStrategy,
        ProcessingStrategy.MERGED: MergedEmbeddingStrategy,
        ProcessingStrategy.COMBINED: CombinedEmbeddingStrategy
    }
    return processors[strategy](paragraph_service, vector_store)

def process_file_content(content: bytes, file_ext: str) -> str:
    """Process file content based on file type."""
    if not content:
        raise ValueError("Empty text")
        
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

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    strategy: ProcessingStrategy = ProcessingStrategy.PARAGRAPHS,
    paragraph_service: ParagraphService = Depends(get_paragraph_service),
    vector_store: PersistentStore = Depends(get_persistent_store)
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
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )

        try:
            # Process content based on file type
            processed_content = process_file_content(content, file_ext)

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
            except (ValueError, RuntimeError) as e:
                logger.error(f"Error processing document {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document: {str(e)}"
                )
        except ValueError as e:
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
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
            "max_file_size_mb": float(MAX_FILE_SIZE_MB),  # Convert to float
            "processing_strategies": [s.value for s in ProcessingStrategy]
        }
    )
