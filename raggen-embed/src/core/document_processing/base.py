"""Base classes and protocols for document processing."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol, Tuple
import numpy as np

from core.text_splitting.service import TextSplitterService
from core.vector_store.base import VectorStore

class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, text_splitter: TextSplitterService, vector_store: VectorStore):
        """
        Initialize document processor.
        
        Args:
            text_splitter: Service for text splitting and embedding
            vector_store: Vector store for saving embeddings
        """
        self.text_splitter = text_splitter
        self.vector_store = vector_store
    
    @abstractmethod
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process document text and return results.
        
        Args:
            text: Document text to process
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    def process_text(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Process text into paragraphs and embeddings.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (paragraphs, embeddings)
        """
        paragraphs = self.text_splitter.split_text(text)
        embeddings = self.text_splitter.get_embeddings(text)
        return paragraphs, embeddings

class EmbeddingMerger(Protocol):
    """Protocol for embedding merging capability."""
    
    def merge_embeddings(self, text_splitter: TextSplitterService, embeddings: np.ndarray) -> np.ndarray:
        """
        Merge multiple embeddings into one.
        
        Args:
            text_splitter: Text splitter service for merging
            embeddings: Embeddings to merge
            
        Returns:
            Merged embedding
        """
        ...

class VectorStorer(Protocol):
    """Protocol for vector storage capability."""
    
    def store_vectors(self, vector_store: VectorStore, vectors: np.ndarray) -> List[int]:
        """
        Store vectors and return their IDs.
        
        Args:
            vector_store: Vector store to use
            vectors: Vectors to store
            
        Returns:
            List of vector IDs
        """
        ...
    
    def store_single_vector(self, vector_store: VectorStore, vector: np.ndarray) -> int:
        """
        Store a single vector and return its ID.
        
        Args:
            vector_store: Vector store to use
            vector: Vector to store
            
        Returns:
            Vector ID
        """
        ...