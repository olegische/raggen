from typing import Dict, Optional
from enum import Enum

from .base import VectorStore
from .faiss_store import FAISSVectorStore
from config.settings import Settings

class VectorStoreType(Enum):
    FAISS = "faiss"
    PERSISTENT = "persistent"

class VectorStoreFactory:
    _instances: Dict[str, VectorStore] = {}

    @classmethod
    def create(cls, store_type: VectorStoreType, settings: Settings, force_new: bool = False) -> VectorStore:
        """
        Create or retrieve a vector store instance.

        Args:
            store_type: Type of vector store to create
            settings: Settings instance
            force_new: If True, always create a new instance

        Returns:
            VectorStore instance
        """
        key = f"{store_type.value}_{settings.faiss_index_path}"

        if not force_new and key in cls._instances:
            return cls._instances[key]

        if store_type == VectorStoreType.FAISS:
            instance = FAISSVectorStore(settings)
        elif store_type == VectorStoreType.PERSISTENT:
            # Импортируем здесь, чтобы избежать циклической зависимости
            from .persistent_store import PersistentStore
            instance = PersistentStore(settings)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

        cls._instances[key] = instance
        return instance

    @classmethod
    def get_or_create(cls, store_type: VectorStoreType, settings: Settings) -> VectorStore:
        """
        Get an existing vector store instance or create a new one if it doesn't exist.

        Args:
            store_type: Type of vector store
            settings: Settings instance

        Returns:
            VectorStore instance
        """
        return cls.create(store_type, settings, force_new=False)

    @classmethod
    def clear_cache(cls):
        """Clear the cache of vector store instances."""
        cls._instances.clear()