import os
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime
import warnings

from .faiss_store import FAISSVectorStore
from config.settings import Settings
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

class PersistentFAISSStore:
    """Persistent vector store implementation that wraps FAISSVectorStore."""
    
    def __init__(self, dimension: int = settings.vector_dim, auto_save: bool = True, index_path: Optional[str] = None):
        """
        Initialize persistent FAISS vector store.
        
        Args:
            dimension: Vector dimension (default: from settings)
            auto_save: Whether to automatically save after modifications
            index_path: Path to the index file (default: from settings)
        """
        self.store_dir = os.path.dirname(index_path or settings.faiss_index_path)
        self.index_path = index_path or settings.faiss_index_path
        self.auto_save = auto_save
        
        # Create directory if it doesn't exist
        os.makedirs(self.store_dir, exist_ok=True)
        
        # Try to load existing index or create new one
        if os.path.exists(self.index_path):
            logger.info("Loading existing index from %s", self.index_path)
            self.store = FAISSVectorStore.load(self.index_path)
            if self.store.dimension != dimension:
                logger.warning(
                    "Loaded index dimension (%d) differs from requested (%d)",
                    self.store.dimension,
                    dimension
                )
                warnings.warn(
                    f"Loaded index dimension ({self.store.dimension}) differs from requested ({dimension})",
                    UserWarning
                )
        else:
            logger.info("Creating new index with dimension %d", dimension)
            self.store = FAISSVectorStore(dimension=dimension)
            if self.auto_save:
                self.store.save(self.index_path)
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the underlying store."""
        self.store.train(vectors)
        if self.auto_save:
            self._save_with_backup()
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> List[int]:
        """Add vectors to the store."""
        result = self.store.add_vectors(vectors, ids)
        if self.auto_save:
            self._save_with_backup()
        return result
    
    def search(self, query_vectors: np.ndarray, k: int = settings.n_results) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        return self.store.search(query_vectors, k)
    
    def _save_with_backup(self) -> None:
        """Save the index with backup."""
        backup_path = None
        
        # Create backup of existing index
        if os.path.exists(self.index_path):
            backup_name = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.faiss"
            backup_path = os.path.join(self.store_dir, backup_name)
            try:
                os.rename(self.index_path, backup_path)
                logger.info("Created backup at %s", backup_path)
            except Exception as e:
                logger.error("Failed to create backup: %s", str(e))
                backup_path = None
                # Continue with save even if backup fails
        
        # Save new index
        try:
            self.store.save(self.index_path)
            logger.info("Index saved successfully")
            
            # Cleanup old backups (keep last 5)
            self._cleanup_old_backups()
        except Exception as e:
            logger.error("Failed to save index: %s", str(e))
            # If save fails and we have a backup, restore it
            if backup_path and os.path.exists(backup_path):
                try:
                    os.rename(backup_path, self.index_path)
                    logger.info("Restored backup after failed save")
                except Exception as restore_error:
                    logger.error("Failed to restore backup: %s", str(restore_error))
            raise  # Re-raise the original error
    
    def _cleanup_old_backups(self, keep_last: int = 5) -> None:
        """Clean up old backup files, keeping the specified number of most recent ones."""
        try:
            # List all backup files
            backup_files = [
                f for f in os.listdir(self.store_dir)
                if f.startswith("index_") and f.endswith(".faiss")
            ]
            
            # Sort by name (which includes timestamp)
            backup_files.sort(reverse=True)
            
            # Remove old backups
            for old_backup in backup_files[keep_last:]:
                try:
                    os.remove(os.path.join(self.store_dir, old_backup))
                    logger.info("Removed old backup: %s", old_backup)
                except Exception as e:
                    logger.error("Failed to remove old backup %s: %s", old_backup, str(e))
        except Exception as e:
            logger.error("Failed to cleanup old backups: %s", str(e))
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return len(self.store)
