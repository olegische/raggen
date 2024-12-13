import os
from typing import List, Optional, Tuple
import time

import numpy as np
import faiss

from config.settings import Settings, IndexType
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

class FAISSVectorStore:
    """
    Vector store implementation using FAISS.
    
    This class provides a simple interface for storing and searching vectors using FAISS.
    The type of index used is configured through settings (flat_l2, ivf_flat, ivf_pq, hnsw_flat).
    Training (if required) is handled automatically when adding vectors.
    """
    
    def __init__(self, dimension: int = settings.vector_dim, index_type: Optional[IndexType] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension (default: from settings)
            index_type: Type of FAISS index to use (default: from settings)
        """
        logger.info("Initializing FAISS vector store with dimension: %d", dimension)
        self.dimension = dimension
        self.index_type = index_type or settings.faiss_index_type
        
        # Create index based on settings
        self.index = self._create_index()
        
        # Track number of vectors and training state
        self.n_vectors = 0
        self.is_trained = False  # All indices start untrained
        
        logger.info("FAISS index created successfully: %s", self.index_type)

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on settings."""
        if self.index_type == IndexType.FLAT_L2:
            logger.info("Creating FlatL2 index for exact search")
            return faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == IndexType.IVF_FLAT:
            logger.info("Creating IVF_FLAT index with %d clusters", settings.n_clusters)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, settings.n_clusters)
            index.nprobe = settings.n_probe
            return index
            
        elif self.index_type == IndexType.IVF_PQ:
            logger.info("Creating IVF_PQ index with %d clusters, M=%d, bits=%d", 
                       settings.n_clusters, settings.pq_m, settings.pq_bits)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, settings.n_clusters, 
                                   settings.pq_m, settings.pq_bits)
            index.nprobe = settings.n_probe
            return index
            
        elif self.index_type == IndexType.HNSW_FLAT:
            logger.info("Creating HNSW index with M=%d", settings.hnsw_m)
            index = faiss.IndexHNSWFlat(self.dimension, settings.hnsw_m)
            index.hnsw.efConstruction = settings.hnsw_ef_construction
            index.hnsw.efSearch = settings.hnsw_ef_search
            return index
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _train(self, vectors: np.ndarray) -> None:
        """
        Internal method to train the index with sample vectors if required.
        Training is handled automatically when adding vectors.
        
        Args:
            vectors: Sample vectors for training (n_samples, dimension)
        """
        if self.is_trained:
            logger.debug("Index is already trained")
            return
            
        # Validate vector dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        
        if self.index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}:
            logger.info("Training index of type %s", self.index_type)
            self.index.train(vectors)
        else:
            logger.debug("Index type %s doesn't require explicit training", self.index_type)
            
        self.is_trained = True
        logger.info("Index is now ready for use")
    
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Training (if required) is handled automatically.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            
        logger.info("Adding %d vectors to index", len(vectors))
        start_time = time.time()
        
        try:
            # Train index if needed
            if not self.is_trained:
                logger.info("Training index before adding vectors")
                self._train(vectors)
            
            # Add vectors to FAISS index
            self.index.add(vectors)
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
            k: Number of results to return per query
            
        Returns:
            Tuple of (distances, indices)
        """
        # First check if index is ready for search
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        # Validate input parameters
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {query_vectors.shape[1]}")
            
        if k <= 0:
            raise ValueError("k must be positive")
            
        # Handle empty index case
        if self.n_vectors == 0:
            logger.warning("No vectors in index, returning empty results")
            return np.array([]).reshape(query_vectors.shape[0], 0), np.array([]).reshape(query_vectors.shape[0], 0)
            
        # Adjust k if needed
        if k > self.n_vectors:
            logger.warning("Requested more results (%d) than available vectors (%d)", k, self.n_vectors)
            k = max(1, self.n_vectors)  # Ensure k is at least 1 if there are vectors
            
        try:
            # Search in FAISS index
            distances, indices = self.index.search(query_vectors, k)
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
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
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
            instance.is_trained = True  # Loaded indices are always trained
            
            # Determine index type from loaded index
            if isinstance(instance.index, faiss.IndexFlatL2):
                instance.index_type = IndexType.FLAT_L2
            elif isinstance(instance.index, faiss.IndexIVFFlat):
                instance.index_type = IndexType.IVF_FLAT
            elif isinstance(instance.index, faiss.IndexIVFPQ):
                instance.index_type = IndexType.IVF_PQ
            elif isinstance(instance.index, faiss.IndexHNSWFlat):
                instance.index_type = IndexType.HNSW_FLAT
            else:
                logger.warning("Unknown index type loaded, using default")
                instance.index_type = settings.faiss_index_type
            
            load_time = time.time() - start_time
            logger.info("Index loaded successfully in %.2f seconds", load_time)
            
            return instance
            
        except Exception as e:
            logger.error("Failed to load index: %s", str(e))
            raise
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return self.n_vectors
