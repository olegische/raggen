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
        skip_index_creation: bool = False
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            settings: Settings instance (required)
            dimension: Vector dimension (default: from settings)
            index_type: Type of FAISS index to use (default: from settings)
            skip_index_creation: If True, don't create index in __init__ (used by load)
        """
        if not settings:
            raise ValueError("Settings must be provided")
            
        self.instance_id = id(self)
        logger.info("[Instance %x] Initializing FAISS vector store", self.instance_id)
        self.settings = settings
        self.dimension = dimension or self.settings.vector_dim
        self.index_type = index_type or self.settings.faiss_index_type
        
        if not skip_index_creation:
            # Create index based on settings
            self.index = self._create_index()
            # Track number of vectors and training state
            self.n_vectors = 0
            self.is_trained = False  # All indices start untrained
            logger.info("[Instance %x] FAISS index created successfully: %s", self.instance_id, self.index_type)
        else:
            self.index = None
            self.n_vectors = 0
            self.is_trained = False

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on settings."""
        if self.index_type == IndexType.FLAT_L2:
            logger.info("[Instance %x] Creating FlatL2 index for exact search", self.instance_id)
            return faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == IndexType.IVF_FLAT:
            logger.info("[Instance %x] Creating IVF_FLAT index with %d clusters", self.instance_id, self.settings.n_clusters)
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
            logger.info("[Instance %x] Creating IVF_PQ index with %d clusters, M=%d, bits=%d",
                       self.instance_id, self.settings.n_clusters, self.settings.pq_m, self.settings.pq_bits)
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
            logger.info("[Instance %x] Creating HNSW index with M=%d", self.instance_id, self.settings.hnsw_m)
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
        
        logger.info("[Instance %x] Training check for index type: %s", self.instance_id, self.index_type)
        logger.info("[Instance %x] Current training state: %s", self.instance_id, self.is_trained)
        logger.info("[Instance %x] Vector shape: %s", self.instance_id, vectors.shape)
        logger.info("[Instance %x] Index total vectors: %d", self.instance_id, self.index.ntotal)
        logger.info("[Instance %x] Index attributes: %s", self.instance_id, dir(self.index))
        
        # Если индекс уже тренирован, выходим
        if self.is_trained:
            logger.info("[Instance %x] Index is already trained, skipping training", self.instance_id)
            return
            
        # Проверяем, нужна ли тренировка для этого типа индекса
        requires_training = self.index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}
        logger.info("[Instance %x] Index requires training: %s", self.instance_id, requires_training)
        
        if not requires_training:
            logger.info("[Instance %x] Index type %s doesn't require explicit training, marking as trained", self.instance_id, self.index_type)
            self.is_trained = True
            return
            
        # Тренируем индекс
        logger.info("[Instance %x] Starting training for index type %s", self.instance_id, self.index_type)
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
        logger.info("[Instance %x] Index training completed successfully", self.instance_id)
        
        # Log index parameters after training
        if hasattr(self.index, 'nprobe'):
            logger.info("[Instance %x] IVF nprobe: %d", self.instance_id, self.index.nprobe)
        if hasattr(self.index, 'pq'):
            logger.info("[Instance %x] PQ M: %d, bits: %d", self.instance_id, self.index.pq.M, self.index.pq.nbits)
        if hasattr(self.index, 'is_trained'):
            logger.info("[Instance %x] Index is_trained flag: %s", self.instance_id, self.index.is_trained)
    
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index. Training (if required) is handled automatically.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            
        logger.info("[Instance %x] Adding %d vectors to index", self.instance_id, len(vectors))
        logger.info("[Instance %x] Vectors adding for index type: %s", self.instance_id, self.index_type)
        start_time = time.time()
        
        try:
            # Train index if needed
            if not self.is_trained:
                logger.info("[Instance %x] Training index before adding vectors", self.instance_id)
                self._train(vectors)
            
            # Add vectors to FAISS index
            self.index.add(vectors)
            self.n_vectors += len(vectors)
            
            add_time = time.time() - start_time
            logger.info("[Instance %x] Vectors added successfully in %.2f seconds", self.instance_id, add_time)
            logger.info("[Instance %x] Vectors added for index type: %s", self.instance_id, self.index_type)
            
        except Exception as e:
            logger.error("[Instance %x] Failed to add vectors: %s", self.instance_id, str(e))
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
            logger.warning("[Instance %x] No vectors in index, returning empty results", self.instance_id)
            return np.array([]).reshape(query_vectors.shape[0], 0), np.array([]).reshape(query_vectors.shape[0], 0)
            
        # Adjust k if needed
        if k > self.n_vectors:
            logger.warning("[Instance %x] Requested more results (%d) than available vectors (%d)", self.instance_id, k, self.n_vectors)
            k = max(1, self.n_vectors)  # Ensure k is at least 1 if there are vectors
            
        try:
            # Search in FAISS index
            logger.info("[Instance %x] Searching vectors index type: %s", self.instance_id, self.index_type)
            distances, indices = self.index.search(query_vectors, k)
            return distances, indices
            
        except Exception as e:
            logger.error("[Instance %x] Failed to search vectors: %s", self.instance_id, str(e))
            raise
    
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        logger.info("[Instance %x] Saving index to: %s", self.instance_id, path)
        start_time = time.time()
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Save the index
            faiss.write_index(self.index, path)
            
            save_time = time.time() - start_time
            logger.info("[Instance %x] Index saved successfully in %.2f seconds", self.instance_id, save_time)
            
        except Exception as e:
            logger.error("[Instance %x] Failed to save index: %s", self.instance_id, str(e))
            raise
    
    @classmethod
    def load(cls, path: str, settings: Settings) -> 'FAISSVectorStore':
        """
        Load an index from disk.
        
        Args:
            path: Path to load the index from
            settings: Settings instance
            
        Returns:
            Loaded vector store
        
        Raises:
            ValueError: If path or settings is None
        """
        if not path:
            raise ValueError("Path must be provided")
        if not settings:
            raise ValueError("Settings must be provided")
        
        logger.info("Loading index from: %s", path)
        start_time = time.time()
        
        try:
            # Load the index first
            loaded_index = faiss.read_index(path)
            
            # Determine index type from loaded index
            index_type = None
            if isinstance(loaded_index, faiss.IndexFlat):
                # Check if it's an L2 index (metric_type == 1)
                if not hasattr(loaded_index, 'metric_type') or loaded_index.metric_type == 1:
                    index_type = IndexType.FLAT_L2
                    logger.info("Detected IndexFlat with L2 metric, treating as FLAT_L2")
                else:
                    logger.warning("Detected IndexFlat with non-L2 metric, this is not supported")
                    raise ValueError("Only L2 metric is supported for flat indices")
            elif isinstance(loaded_index, faiss.IndexFlatL2):
                index_type = IndexType.FLAT_L2
                logger.info("Detected FLAT_L2 index (exact search, no training needed)")
            elif isinstance(loaded_index, faiss.IndexIVFFlat):
                index_type = IndexType.IVF_FLAT
                logger.info("Detected IVF_FLAT index (approximate search with clustering)")
            elif isinstance(loaded_index, faiss.IndexIVFPQ):
                index_type = IndexType.IVF_PQ
                logger.info("Detected IVF_PQ index (compressed vectors with clustering)")
            elif isinstance(loaded_index, faiss.IndexHNSWFlat):
                index_type = IndexType.HNSW_FLAT
                logger.info("Detected HNSW_FLAT index (graph-based search)")
            else:
                logger.error("Unsupported index type loaded: %s", type(loaded_index).__name__)
                raise ValueError(f"Unsupported index type: {type(loaded_index).__name__}")
            
            # Create instance without creating a new index
            instance = cls(settings=settings, dimension=loaded_index.d, index_type=index_type, skip_index_creation=True)
            instance.index = loaded_index
            instance.dimension = loaded_index.d
            instance.n_vectors = loaded_index.ntotal
            # Log detailed index information
            logger.info("[Instance %x] Loading index from path: %s", instance.instance_id, path)
            logger.info("[Instance %x] Loaded index type: %s", instance.instance_id, type(instance.index).__name__)
            logger.info("[Instance %x] Index attributes: %s", instance.instance_id, dir(instance.index))
            logger.info("[Instance %x] Index dimension: %d", instance.instance_id, instance.dimension)
            logger.info("[Instance %x] Index total vectors: %d", instance.instance_id, instance.n_vectors)
            
            # Для FLAT_L2 индекса всегда устанавливаем is_trained=True
            if instance.index_type == IndexType.FLAT_L2:
                instance.is_trained = True
                logger.info("[Instance %x] FLAT_L2 index is always trained", instance.instance_id)
            else:
                # Для других типов индексов проверяем наличие векторов
                instance.is_trained = instance.n_vectors > 0
                logger.info("[Instance %x] Setting training state to %s based on vectors count",
                           instance.instance_id, instance.is_trained)
            
            load_time = time.time() - start_time
            logger.info("[Instance %x] Index loaded successfully in %.2f seconds", instance.instance_id, load_time)
            
            return instance
            
        except Exception as e:
            logger.error("Failed to load index: %s", str(e))
            raise
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return self.n_vectors