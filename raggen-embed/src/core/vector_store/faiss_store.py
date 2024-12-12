import os
from typing import List, Optional, Tuple, Dict
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
        
        # Track number of vectors
        self.n_vectors = 0
        self.is_trained = False  # Even though FlatL2 doesn't need training, we keep this for consistency
        
        # Track user-facing IDs
        self.next_user_id = 1  # Start user IDs from 1
        self.internal_to_user_ids: Dict[int, int] = {}  # Map internal FAISS IDs to user IDs
        self.user_to_internal_ids: Dict[int, int] = {}  # Map user IDs to internal FAISS IDs
        
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
            
        # Validate vector dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        
        # For FlatL2 we don't actually need training, but we keep this for API consistency
        self.is_trained = True
        logger.info("Index trained successfully")
    
    def _get_next_user_ids(self, count: int) -> List[int]:
        """Get next available user IDs."""
        user_ids = list(range(self.next_user_id, self.next_user_id + count))
        self.next_user_id += count
        return user_ids
    
    def _map_internal_to_user_ids(self, internal_ids: np.ndarray) -> np.ndarray:
        """Map internal FAISS IDs to user-facing IDs."""
        # Handle 2D array from FAISS search results
        result = np.zeros_like(internal_ids)
        for i in range(internal_ids.shape[0]):
            for j in range(internal_ids.shape[1]):
                internal_id = int(internal_ids[i, j])
                result[i, j] = self.internal_to_user_ids.get(internal_id, internal_id + 1)
        return result
    
    def _map_user_to_internal_ids(self, user_ids: List[int]) -> List[int]:
        """Map user-facing IDs to internal FAISS IDs."""
        return [self.user_to_internal_ids.get(i, -1) for i in user_ids]
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add (n_vectors, dimension)
            ids: Optional user-provided vector IDs (n_vectors,)
            
        Returns:
            List of assigned vector IDs (user-facing)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
            
        logger.info("Adding %d vectors to index", len(vectors))
        start_time = time.time()
        
        try:
            # Generate or validate user IDs
            if ids is None:
                user_ids = self._get_next_user_ids(len(vectors))
            else:
                if len(ids) != len(vectors):
                    raise ValueError("Number of IDs must match number of vectors")
                user_ids = ids
            
            # Get internal IDs that will be assigned by FAISS
            internal_ids = list(range(self.n_vectors, self.n_vectors + len(vectors)))
            
            # Update ID mappings
            for internal_id, user_id in zip(internal_ids, user_ids):
                self.internal_to_user_ids[internal_id] = user_id
                self.user_to_internal_ids[user_id] = internal_id
            
            # Add vectors to FAISS index
            self.index.add(vectors)
            self.n_vectors += len(vectors)
            
            add_time = time.time() - start_time
            logger.info("Vectors added successfully in %.2f seconds", add_time)
            
            return user_ids
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
            Tuple of (distances, user_facing_indices)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {query_vectors.shape[1]}")
            
        if k <= 0:
            raise ValueError("k must be positive")
            
        if k > self.n_vectors:
            logger.warning("Requested more results (%d) than available vectors (%d)", k, self.n_vectors)
            k = max(1, self.n_vectors)  # Ensure k is at least 1 if there are vectors
            
        try:
            if self.n_vectors == 0:
                # If no vectors, return empty arrays
                return np.array([]).reshape(query_vectors.shape[0], 0), np.array([]).reshape(query_vectors.shape[0], 0)
            
            # Get internal FAISS results
            distances, internal_indices = self.index.search(query_vectors, k)
            
            # Map internal indices to user-facing IDs
            user_indices = self._map_internal_to_user_ids(internal_indices)
            
            return distances, user_indices
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
            
            # Save ID mappings
            mappings_path = path + ".mappings"
            try:
                # Log current state
                logger.info("Current mappings state:")
                logger.info("next_user_id: %d", self.next_user_id)
                logger.info("internal_to_user_ids: %s", dict(self.internal_to_user_ids))
                logger.info("user_to_internal_ids: %s", dict(self.user_to_internal_ids))
                
                # Convert dictionaries to arrays
                internal_to_user = np.array([(k, v) for k, v in self.internal_to_user_ids.items()],
                                         dtype=[('key', 'i4'), ('value', 'i4')])
                user_to_internal = np.array([(k, v) for k, v in self.user_to_internal_ids.items()],
                                         dtype=[('key', 'i4'), ('value', 'i4')])
                
                # Prepare data for saving
                mappings_data = {
                    'next_user_id': np.array([self.next_user_id], dtype=np.int32),
                    'internal_to_user': internal_to_user,
                    'user_to_internal': user_to_internal
                }
                
                # Log data being saved
                logger.info("Saving mappings data: %s", mappings_data)
                logger.info("Saving to path: %s", mappings_path)
                
                # Save mappings
                np.savez(mappings_path, **mappings_data)
                
                # Wait for file system
                time.sleep(0.1)
                
                # Check if file exists (with .npz extension) and log its properties
                npz_path = mappings_path + '.npz'
                if os.path.exists(npz_path):
                    logger.info("Mappings file created successfully")
                    logger.info("File size: %d bytes", os.path.getsize(npz_path))
                else:
                    logger.error("Mappings file not found after saving")
                    # Use '.' for current directory if dirname is empty
                    dir_path = os.path.dirname(path) or '.'
                    logger.error("Directory contents: %s", os.listdir(dir_path))
                    raise RuntimeError(f"Failed to create mappings file: {npz_path}")
                
            except Exception as e:
                logger.error("Failed to save mappings: %s", str(e))
                raise
            
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
            
            # Load ID mappings
            mappings_path = path + ".mappings.npz"  # Add .npz extension here
            if os.path.exists(mappings_path):
                logger.info("Loading mappings from %s", mappings_path)
                try:
                    mappings = np.load(mappings_path, allow_pickle=True)
                    
                    # Load next_user_id
                    instance.next_user_id = int(mappings["next_user_id"][0])
                    
                    # Load mappings from structured arrays
                    internal_to_user = mappings["internal_to_user"]
                    user_to_internal = mappings["user_to_internal"]
                    
                    # Convert structured arrays to dictionaries
                    instance.internal_to_user_ids = {int(x["key"]): int(x["value"])
                                                   for x in internal_to_user}
                    instance.user_to_internal_ids = {int(x["key"]): int(x["value"])
                                                   for x in user_to_internal}
                    
                    logger.info("Loaded mappings: next_user_id=%d, mappings=%s",
                              instance.next_user_id, instance.internal_to_user_ids)
                except Exception as e:
                    logger.error("Failed to load mappings: %s", str(e))
                    logger.warning("Falling back to default sequential IDs")
                    instance._init_default_mappings()
            else:
                logger.warning("No mappings file found at %s, using default sequential IDs", mappings_path)
                instance._init_default_mappings()
            
            load_time = time.time() - start_time
            logger.info("Index loaded successfully in %.2f seconds", load_time)
            
            return instance
        except Exception as e:
            logger.error("Failed to load index: %s", str(e))
            raise
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return self.n_vectors
