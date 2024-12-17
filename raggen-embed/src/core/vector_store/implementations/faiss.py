"""FAISS vector store implementation."""
import os
from typing import Optional, Tuple
import time

import numpy as np
import faiss

from ..base import VectorStore
from config.settings import Settings, IndexType
from utils.logging import get_logger

logger = get_logger(__name__)

class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS."""
    
    def __init__(
        self,
        settings: Settings,
        dimension: Optional[int] = None,
        index_type: Optional[IndexType] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            settings: Settings instance (required)
            dimension: Vector dimension (default: from settings)
            index_type: Type of FAISS index to use (default: from settings)
        """
        if not settings:
            raise ValueError("Settings must be provided")
            
        logger.info("Initializing FAISS vector store")
        self.settings = settings
        self.dimension = dimension or self.settings.vector_dim
        self.index_type = index_type or self.settings.faiss_index_type
        
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
            logger.info("Creating IVF_FLAT index with %d clusters", self.settings.n_clusters)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.settings.n_clusters)
            index.nprobe = self.settings.n_probe
            # Set verbose=False for all components
            if hasattr(index, 'verbose'):
                index.verbose = False
            if hasattr(index.quantizer, 'verbose'):
                index.quantizer.verbose = False
            return index
            
        elif self.index_type == IndexType.IVF_PQ:
            logger.info("Creating IVF_PQ index with %d clusters, M=%d, bits=%d",
                       self.settings.n_clusters, self.settings.pq_m, self.settings.pq_bits)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, self.settings.n_clusters,
                                   self.settings.pq_m, self.settings.pq_bits)
            index.nprobe = self.settings.n_probe
            # Set verbose=False for all components
            if hasattr(index, 'verbose'):
                index.verbose = False
            if hasattr(index.quantizer, 'verbose'):
                index.quantizer.verbose = False
            if hasattr(index, 'pq') and hasattr(index.pq, 'verbose'):
                index.pq.verbose = False
            return index
            
        elif self.index_type == IndexType.HNSW_FLAT:
            logger.info("Creating HNSW index with M=%d", self.settings.hnsw_m)
            index = faiss.IndexHNSWFlat(self.dimension, self.settings.hnsw_m)
            index.hnsw.efConstruction = self.settings.hnsw_ef_construction
            index.hnsw.efSearch = self.settings.hnsw_ef_search
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
        # Validate vector dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        
        logger.info("Training check for index type: %s", self.index_type)
        logger.info("Current training state: %s", self.is_trained)
        logger.info("Vector shape: %s", vectors.shape)
        logger.info("Index total vectors: %d", self.index.ntotal)
        logger.info("Index attributes: %s", dir(self.index))
        
        # Если индекс уже тренирован, выходим
        if self.is_trained:
            logger.info("Index is already trained, skipping training")
            return
            
        # Проверяем, нужна ли тренировка для этого типа индекса
        requires_training = self.index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}
        logger.info("Index requires training: %s", requires_training)
        
        if not requires_training:
            logger.info("Index type %s doesn't require explicit training, marking as trained", self.index_type)
            self.is_trained = True
            return
            
        # Тренируем индекс
        logger.info("Starting training for index type %s", self.index_type)
        # Set verbose=False for all components before training
        if hasattr(self.index, 'verbose'):
            self.index.verbose = False
        if hasattr(self.index.quantizer, 'verbose'):
            self.index.quantizer.verbose = False
        if hasattr(self.index, 'pq') and hasattr(self.index.pq, 'verbose'):
            self.index.pq.verbose = False
        if hasattr(self.index, 'cp'):
            self.index.cp.verbose = False
        self.index.train(vectors)
        self.is_trained = True
        logger.info("Index training completed successfully")
        
        # Log index parameters after training
        if hasattr(self.index, 'nprobe'):
            logger.info("IVF nprobe: %d", self.index.nprobe)
        if hasattr(self.index, 'pq'):
            logger.info("PQ M: %d, bits: %d", self.index.pq.M, self.index.pq.nbits)
        if hasattr(self.index, 'is_trained'):
            logger.info("Index is_trained flag: %s", self.index.is_trained)
    
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
    
    def search(self, query_vectors: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of results to return per query (default: from settings)
            
        Returns:
            Tuple of (distances, indices)
        """
        # Use settings.n_results as default if k not provided
        k = k if k is not None else self.settings.n_results
        
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
    def load(cls, path: Optional[str] = None, settings: Optional[Settings] = None) -> 'FAISSVectorStore':
        """
        Load an index from disk.
        
        Args:
            path: Optional path to load the index from
            settings: Optional Settings instance
            
        Returns:
            Loaded vector store
        
        Raises:
            ValueError: If both path and settings are None
        """
        if path is None and settings is None:
            raise ValueError("Either path or settings must be provided")
        
        if settings is None:
            # Если settings не предоставлены, создаем экземпляр с дефолтными настройками
            settings = Settings()
        
        if path is None:
            # Если path не предоставлен, используем значение из settings
            path = settings.faiss_index_path
        
        logger.info("Loading index from: %s", path)
        start_time = time.time()
        
        try:
            # Create instance with settings
            instance = cls(settings=settings)
            
            # Load the index
            instance.index = faiss.read_index(path)
            instance.dimension = instance.index.d
            instance.n_vectors = instance.index.ntotal
            # Log detailed index information
            logger.info("Loading index from path: %s", path)
            logger.info("Loaded index type: %s", type(instance.index).__name__)
            logger.info("Index attributes: %s", dir(instance.index))
            logger.info("Index dimension: %d", instance.index.d)
            logger.info("Index total vectors: %d", instance.index.ntotal)
            
            # Determine index type from loaded index
            if isinstance(instance.index, faiss.IndexFlatL2):
                instance.index_type = IndexType.FLAT_L2
                logger.info("Detected FLAT_L2 index (exact search, no training needed)")
            elif isinstance(instance.index, faiss.IndexIVFFlat):
                instance.index_type = IndexType.IVF_FLAT
                logger.info("Detected IVF_FLAT index (approximate search with clustering)")
                if hasattr(instance.index, 'nprobe'):
                    logger.info("IVF nprobe: %d", instance.index.nprobe)
            elif isinstance(instance.index, faiss.IndexIVFPQ):
                instance.index_type = IndexType.IVF_PQ
                logger.info("Detected IVF_PQ index (compressed vectors with clustering)")
                if hasattr(instance.index, 'pq'):
                    logger.info("PQ M: %d, bits: %d", instance.index.pq.M, instance.index.pq.nbits)
            elif isinstance(instance.index, faiss.IndexHNSWFlat):
                instance.index_type = IndexType.HNSW_FLAT
                logger.info("Detected HNSW_FLAT index (graph-based search)")
                if hasattr(instance.index, 'hnsw'):
                    logger.info("HNSW M: %d, efConstruction: %d, efSearch: %d",
                              instance.index.hnsw.M,
                              instance.index.hnsw.efConstruction,
                              instance.index.hnsw.efSearch)
            else:
                logger.warning("Unknown index type loaded: %s, using default: %s",
                             type(instance.index).__name__,
                             instance.settings.faiss_index_type)
                instance.index_type = instance.settings.faiss_index_type
            
            # Set training state based on index type
            requires_training = instance.index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}
            if requires_training:
                # Для индексов, требующих тренировки, проверяем состояние
                instance.is_trained = getattr(instance.index, 'is_trained', False)
            else:
                # Индексы без тренировки всегда готовы к использованию
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