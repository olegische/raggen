import os
from typing import List, Optional, Tuple
import time

import numpy as np
import faiss

from config.settings import Settings
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

class FAISSVectorStore:
    """Vector store implementation using FAISS."""
    
    def __init__(self, dimension: int = settings.vector_dim):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension (default: from settings)
        """
        logger.info("Initializing FAISS vector store with dimension: %d", dimension)
        self.dimension = dimension
        
        # Create a flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(dimension)
        # Add IVF for faster search with slight accuracy trade-off
        self.index = faiss.IndexIVFFlat(self.index, dimension, settings.n_clusters)
        
        # Track number of vectors
        self.n_vectors = 0
        self.is_trained = False
        
        logger.info("FAISS index created successfully")
    
    def train(self, vectors: np.ndarray) -> None:
        """
        Train the index with sample vectors.
        
        Args:
            vectors: Sample vectors for training (n_samples, dimension)
        """
        if self.is_trained:
            logger.warning("Index is already trained")
            return
            
        # FAISS IVF requires at least 39 * n_clusters points for training
        min_points = 39 * settings.n_clusters
        if len(vectors) < min_points:
            raise ValueError(f"Need at least {min_points} vectors for training, got {len(vectors)}")
            
        logger.info("Training FAISS index with %d vectors", len(vectors))
        start_time = time.time()
        
        try:
            self.index.train(vectors)
            self.is_trained = True
            train_time = time.time() - start_time
            logger.info("Index trained successfully in %.2f seconds", train_time)
        except Exception as e:
            logger.error("Failed to train index: %s", str(e))
            raise
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
            ids: Optional vector IDs (n_vectors,)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            
        logger.info("Adding %d vectors to index", len(vectors))
        start_time = time.time()
        
        try:
            if ids is None:
                # Generate sequential IDs
                ids = np.arange(self.n_vectors, self.n_vectors + len(vectors))
            
            self.index.add_with_ids(vectors, ids)
            self.n_vectors += len(vectors)
            add_time = time.time() - start_time
            logger.info("Vectors added successfully in %.2f seconds", add_time)
        except Exception as e:
            logger.error("Failed to add vectors: %s", str(e))
            raise
    
    def search(self, query_vectors: np.ndarray, k: int = settings.n_results) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {query_vectors.shape[1]}")
            
        if k > self.n_vectors:
            logger.warning("Requested more results (%d) than available vectors (%d)", k, self.n_vectors)
            k = min(k, self.n_vectors)
            
        logger.info("Searching for %d similar vectors", k)
        start_time = time.time()
        
        try:
            # Set number of clusters to probe (trade-off between speed and accuracy)
            self.index.nprobe = min(settings.n_probe, settings.n_clusters)
            distances, indices = self.index.search(query_vectors, k)
            
            search_time = time.time() - start_time
            logger.info("Search completed in %.2f seconds", search_time)
            
            return distances, indices
        except Exception as e:
            logger.error("Failed to search vectors: %s", str(e))
            raise
    
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        logger.info("Saving index to: %s", path)
        start_time = time.time()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the index
            faiss.write_index(self.index, path)
            
            save_time = time.time() - start_time
            logger.info("Index saved successfully in %.2f seconds", save_time)
        except Exception as e:
            logger.error("Failed to save index: %s", str(e))
            raise
    
    @classmethod
    def load(cls, path: str) -> 'FAISSVectorStore':
        """
        Load an index from disk.
        
        Args:
            path: Path to load the index from
            
        Returns:
            Loaded vector store
        """
        logger.info("Loading index from: %s", path)
        start_time = time.time()
        
        try:
            # Create instance
            instance = cls()
            
            # Load the index
            instance.index = faiss.read_index(path)
            instance.dimension = instance.index.d
            instance.n_vectors = instance.index.ntotal
            instance.is_trained = True
            
            load_time = time.time() - start_time
            logger.info("Index loaded successfully in %.2f seconds", load_time)
            
            return instance
        except Exception as e:
            logger.error("Failed to load index: %s", str(e))
            raise
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return self.n_vectors
