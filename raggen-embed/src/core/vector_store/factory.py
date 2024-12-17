"""Factory for creating vector store instances."""
from typing import Dict, Type
from enum import Enum

from .base import VectorStore
from .implementations import FAISSVectorStore, PersistentStore
from config.settings import Settings

class VectorStoreType(Enum):
    """Available vector store types."""
    FAISS = "faiss"
    PERSISTENT = "persistent"

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    _implementations: Dict[str, Type[VectorStore]] = {
        VectorStoreType.FAISS.value: FAISSVectorStore,
        VectorStoreType.PERSISTENT.value: PersistentStore
    }
    
    @classmethod
    def create(cls, store_type: VectorStoreType, settings: Settings) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store to create
            settings: Settings instance
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If store type is unknown
        """
        if not isinstance(store_type, VectorStoreType):
            raise ValueError(f"Unknown store type: {store_type}")
            
        implementation = cls._implementations[store_type.value]
        return implementation(settings)
    
    @classmethod
    def register_implementation(cls, name: str, implementation: Type[VectorStore]) -> None:
        """
        Register a new vector store implementation.
        
        Args:
            name: Name for the implementation
            implementation: Class implementing VectorStore
        """
        cls._implementations[name] = implementation