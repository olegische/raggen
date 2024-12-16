import os
from typing import Optional, Tuple
from datetime import datetime
import warnings
import numpy as np

from .base import VectorStore
from .faiss_store import FAISSVectorStore
from config.settings import Settings
from utils.logging import get_logger

logger = get_logger(__name__)

class PersistentStore(VectorStore):
    """Persistent vector store implementation that wraps any VectorStore."""
    
    def __init__(
        self,
        settings: Settings,
        store: Optional[VectorStore] = None,
        auto_save: bool = True,
    ):
        """
        Initialize persistent vector store.
        
        Args:
            settings: Settings instance (required)
            store: Vector store instance to wrap (default: new FAISSVectorStore)
            auto_save: Whether to automatically save after modifications
            
        Raises:
            ValueError: If settings is not provided
            TypeError: If store is provided but not a VectorStore instance
            RuntimeError: If directory creation or access fails
        """
        if not settings:
            raise ValueError("Settings must be provided")
            
        super().__init__()
        logger.info("[PersistentStore] Starting initialization")
        
        self.settings = settings
        logger.info("[PersistentStore] Using FAISS_INDEX_PATH: %s", self.settings.faiss_index_path)
        
        # Set index path from settings
        self.index_path = self.settings.faiss_index_path
        self.store_dir = os.path.dirname(self.index_path)
        logger.info("[PersistentStore] Derived store_dir: %s", self.store_dir)
        
        self.auto_save = auto_save
        logger.info("[PersistentStore] Auto-save enabled: %s", self.auto_save)
        
        # Create directory if it doesn't exist and check permissions
        try:
            logger.info("[PersistentStore] Creating directory: %s", self.store_dir)
            os.makedirs(self.store_dir, exist_ok=True)
            
            # Check write permissions
            if not os.access(self.store_dir, os.W_OK):
                raise OSError(f"Directory {self.store_dir} is not writable")
        except OSError as e:
            logger.error("[PersistentStore] Failed to initialize store directory: %s", str(e))
            raise RuntimeError(f"Failed to initialize store directory: {e}")
        
        # Validate store type if provided
        if store is not None and not isinstance(store, VectorStore):
            raise TypeError("store must be an instance of VectorStore")
        
        # Try to load existing index or use provided store
        if os.path.exists(self.index_path):
            logger.info("[PersistentStore] Loading existing index from %s", self.index_path)
            self.store = FAISSVectorStore.load(self.index_path, settings=self.settings)
        else:
            logger.info("[PersistentStore] Using provided store or creating new one")
            self.store = store if store is not None else FAISSVectorStore(settings=self.settings)
            if self.auto_save:
                logger.info("[PersistentStore] Auto-saving new store to %s", self.index_path)
                self.store.save(self.index_path)
        
        logger.info("[PersistentStore] Initialization complete")
    
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the store."""
        logger.info("[PersistentStore] Adding vectors")
        self.store.add(vectors)
        if self.auto_save:
            logger.info("[PersistentStore] Auto-saving after adding vectors")
            self._save_with_backup()
    
    def search(self, query_vectors: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        logger.info("[PersistentStore] Searching for vectors")
        k = k if k is not None else self.settings.n_results
        return self.store.search(query_vectors=query_vectors, k=k)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the store to disk with backup.
        
        Args:
            path: Optional override path (default: use path from settings)
            
        Raises:
            OSError: If directory is not writable or disk space issues
            RuntimeError: If save operation fails for other reasons
        """
        logger.info("[PersistentStore] Explicit save requested")
        save_path = path or self.index_path
        self._save_with_backup(save_path)
    
    @classmethod
    def load(cls, path: Optional[str] = None, settings: Optional[Settings] = None) -> 'PersistentStore':
        """
        Load a store from disk.
        
        Args:
            path: Optional override path (default: use path from settings)
            settings: Settings instance (required)
            
        Returns:
            New instance of PersistentStore
            
        Raises:
            ValueError: If settings is not provided
            RuntimeError: If load operation fails
        """
        if not settings:
            raise ValueError("Settings must be provided")
            
        logger.info("[PersistentStore] Loading store")
        
        # If path not provided, use settings
        if not path:
            path = settings.faiss_index_path
            logger.info("[PersistentStore] Using path from settings: %s", path)
        else:
            logger.info("[PersistentStore] Using provided path: %s", path)
            
        # Create store and load index
        store = FAISSVectorStore(settings=settings)
        store.load(path)
        
        # Create persistent store with loaded index
        return cls(store=store, settings=settings)
    
    def _save_with_backup(self, path: Optional[str] = None) -> None:
        """
        Save the index with backup.
        
        Args:
            path: Optional override path (default: use path from settings)
            
        Raises:
            OSError: If directory is not writable or disk space issues
            RuntimeError: If save operation fails for other reasons
        """
        # Check write permissions
        if not os.access(self.store_dir, os.W_OK):
            logger.error("[PersistentStore] Directory %s is not writable", self.store_dir)
            raise OSError(f"Directory {self.store_dir} is not writable")
            
        save_path = path or self.index_path
        logger.info("[PersistentStore] Saving to path: %s", save_path)
        backup_path = None
        
        # Create backup of existing index
        if os.path.exists(save_path):
            backup_name = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.faiss"
            backup_path = os.path.join(self.store_dir, backup_name)
            try:
                os.rename(save_path, backup_path)
                logger.info("[PersistentStore] Created backup at %s", backup_path)
            except OSError as e:
                logger.error("[PersistentStore] Failed to create backup: %s", str(e))
                backup_path = None
                raise  # Пробрасываем OSError
            except Exception as e:
                logger.error("[PersistentStore] Failed to create backup: %s", str(e))
                backup_path = None
        
        # Save new index
        try:
            self.store.save(save_path)
            logger.info("[PersistentStore] Index saved successfully")
            
            # Cleanup old backups
            self._cleanup_old_backups()
        except OSError as e:
            logger.error("[PersistentStore] Failed to save index: %s", str(e))
            # If save fails and we have a backup, restore it
            if backup_path and os.path.exists(backup_path):
                try:
                    os.rename(backup_path, save_path)
                    logger.info("[PersistentStore] Restored backup after failed save")
                except Exception as restore_error:
                    logger.error("[PersistentStore] Failed to restore backup: %s", str(restore_error))
            raise  # Пробрасываем OSError
        except Exception as e:
            logger.error("[PersistentStore] Failed to save index: %s", str(e))
            # If save fails and we have a backup, restore it
            if backup_path and os.path.exists(backup_path):
                try:
                    os.rename(backup_path, save_path)
                    logger.info("[PersistentStore] Restored backup after failed save")
                except Exception as restore_error:
                    logger.error("[PersistentStore] Failed to restore backup: %s", str(restore_error))
            raise RuntimeError(f"Failed to save index: {e}")
    
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
                    logger.info("[PersistentStore] Removed old backup: %s", old_backup)
                except Exception as e:
                    logger.error("[PersistentStore] Failed to remove old backup %s: %s", old_backup, str(e))
        except Exception as e:
            logger.error("[PersistentStore] Failed to cleanup old backups: %s", str(e))
    
    def __len__(self) -> int:
        """Get number of vectors in the store."""
        return len(self.store)
