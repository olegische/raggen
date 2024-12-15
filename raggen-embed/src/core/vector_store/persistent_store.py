import os
from typing import Optional, Tuple
from datetime import datetime
import warnings
import numpy as np

from .base import VectorStore
from .faiss_store import FAISSVectorStore
from config.settings import Settings
from utils.logging import get_logger

settings = Settings()
logger = get_logger(__name__)

class PersistentStore(VectorStore):
    """Persistent vector store implementation that wraps any VectorStore."""
    
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        index_path: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Initialize persistent vector store.
        
        Args:
            store: Vector store instance to wrap (default: new FAISSVectorStore)
            index_path: Path to the index file (default: from settings)
            auto_save: Whether to automatically save after modifications
        """
        self.index_path = index_path or settings.faiss_index_path
        self.store_dir = os.path.dirname(self.index_path)
        self.auto_save = auto_save
        
        # Create directory if it doesn't exist
        os.makedirs(self.store_dir, exist_ok=True)
        
        # Try to load existing index or create new one
        if os.path.exists(self.index_path):
            logger.info("Loading existing index from %s", self.index_path)
            self.store = FAISSVectorStore.load(self.index_path)
        else:
            logger.info("Creating new store")
            self.store = store or FAISSVectorStore()
            if self.auto_save:
                self.store.save(self.index_path)
    
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the store."""
        self.store.add(vectors)
        if self.auto_save:
            self._save_with_backup()
    
    def search(self, query_vectors: np.ndarray, k: int = settings.n_results) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        return self.store.search(query_vectors, k)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the store to disk with backup.
        
        Args:
            path: Path to save to (default: configured index path)
        """
        self._save_with_backup(path)
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'PersistentStore':
        """
        Load a store from disk.
        
        Args:
            path: Path to load from (default: from settings)
            
        Returns:
            New instance of PersistentStore
        """
        return cls(index_path=path)
    
    def _save_with_backup(self, path: Optional[str] = None) -> None:
        """
        Save the index with backup.
        
        Args:
            path: Path to save to (default: configured index path)
        """
        save_path = path or self.index_path
        backup_path = None
        
        # Create backup of existing index
        if os.path.exists(save_path):
            backup_name = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.faiss"
            backup_path = os.path.join(self.store_dir, backup_name)
            try:
                os.rename(save_path, backup_path)
                logger.info("Created backup at %s", backup_path)
            except Exception as e:
                logger.error("Failed to create backup: %s", str(e))
                backup_path = None
        
        # Save new index
        try:
            self.store.save(save_path)
            logger.info("Index saved successfully")
            
            # Cleanup old backups
            self._cleanup_old_backups()
        except Exception as e:
            logger.error("Failed to save index: %s", str(e))
            # If save fails and we have a backup, restore it
            if backup_path and os.path.exists(backup_path):
                try:
                    os.rename(backup_path, save_path)
                    logger.info("Restored backup after failed save")
                except Exception as restore_error:
                    logger.error("Failed to restore backup: %s", str(restore_error))
            raise
    
    def _cleanup_old_backups(self, keep_last: int = 5) -> None:
        """
        Clean up old backup files.
        
        Args:
            keep_last: Number of recent backups to keep
        """
        try:
            backup_files = [
                f for f in os.listdir(self.store_dir)
                if f.startswith("index_") and f.endswith(".faiss")
            ]
            
            backup_files.sort(reverse=True)
            
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
