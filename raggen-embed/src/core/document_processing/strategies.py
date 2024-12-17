"""Document processing strategies."""
import logging
from typing import List, Dict, Any
import numpy as np

from .base import DocumentProcessor, EmbeddingMerger, VectorStorer

logger = logging.getLogger(__name__)

class BaseVectorStorer:
    """Base implementation of vector storing capability."""
    
    @staticmethod
    def store_vectors(vector_store, vectors: np.ndarray) -> List[int]:
        """Store vectors and return their IDs."""
        try:
            logger.info("Storing vectors with shape: %s using store type: %s", 
                       vectors.shape, type(vector_store).__name__)
            vector_store.add(vectors)
            return list(range(len(vectors)))  # Simplified ID generation
        except Exception as e:
            logger.error("Failed to store vectors: %s", str(e))
            raise RuntimeError(f"Failed to store vectors: {e}")
    
    @staticmethod
    def store_single_vector(vector_store, vector: np.ndarray) -> int:
        """Store a single vector and return its ID."""
        try:
            logger.info("Storing single vector with shape: %s using store type: %s", 
                       vector.shape, type(vector_store).__name__)
            vector_store.add(vector.reshape(1, -1))
            return 0  # Simplified ID generation
        except Exception as e:
            logger.error("Failed to store single vector: %s", str(e))
            raise RuntimeError(f"Failed to store single vector: {e}")

class BaseEmbeddingMerger:
    """Base implementation of embedding merging capability."""
    
    def merge_embeddings(self, text_splitter, embeddings: np.ndarray) -> np.ndarray:
        """Merge multiple embeddings into one."""
        return text_splitter.merge_embeddings(embeddings)

class ParagraphEmbeddingStrategy(DocumentProcessor, BaseVectorStorer):
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

class MergedEmbeddingStrategy(DocumentProcessor, BaseEmbeddingMerger, BaseVectorStorer):
    """Strategy for processing document into a single merged embedding."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Process text into paragraphs and embeddings
        paragraphs, embeddings = self.process_text(text)
        
        # Merge embeddings and store
        merged_embedding = self.merge_embeddings(self.text_splitter, embeddings)
        vector_id = self.store_single_vector(self.vector_store, merged_embedding)
        
        return {
            "strategy": "merged",
            "paragraphs_count": len(paragraphs),
            "vector_id": vector_id,
            "paragraphs": paragraphs
        }

class CombinedEmbeddingStrategy(DocumentProcessor, BaseEmbeddingMerger, BaseVectorStorer):
    """Strategy that combines both paragraph and merged embeddings."""
    
    def process(self, text: str) -> Dict[str, Any]:
        # Process text into paragraphs and embeddings
        paragraphs, embeddings = self.process_text(text)
        
        # Store individual paragraph embeddings
        paragraph_vector_ids = self.store_vectors(self.vector_store, embeddings)
        
        # Merge embeddings and store
        merged_embedding = self.merge_embeddings(self.text_splitter, embeddings)
        merged_vector_id = self.store_single_vector(self.vector_store, merged_embedding)
        
        return {
            "strategy": "combined",
            "paragraphs_count": len(paragraphs),
            "paragraph_vector_ids": paragraph_vector_ids,
            "merged_vector_id": merged_vector_id,
            "paragraphs": paragraphs
        }