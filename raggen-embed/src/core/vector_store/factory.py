"""Factory for creating vector store instances."""
from typing import Dict, Type

from .base import VectorStore
from .implementations import FAISSVectorStore, PersistentStore
from config.settings import Settings, VectorStoreType

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
        # If string is provided, try to convert to enum or use as custom type
        if isinstance(store_type, str):
            try:
                store_type = VectorStoreType(store_type)
            except ValueError:
                # Not a standard type, check if it's a registered custom type
                if store_type not in cls._implementations:
                    raise ValueError(f"'{store_type}' is not a valid VectorStoreType")
                return cls._implementations[store_type](settings)
        elif not isinstance(store_type, VectorStoreType):
            raise ValueError(f"Unknown store type: {store_type}")
        
        # Handle standard types
        return cls._implementations[store_type.value](settings)
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