"""Factory for creating vector store instances."""
from typing import Dict, Type, Union, Optional

from .base import VectorStore
from .implementations import FAISSVectorStore, PersistentStore
from config.settings import Settings, VectorStoreServiceType, VectorStoreImplementationType

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    # High-level service implementations
    _service_implementations: Dict[str, Type[VectorStore]] = {
        VectorStoreServiceType.PERSISTENT.value: PersistentStore
    }
    
    # Low-level store implementations
    _store_implementations: Dict[str, Type[VectorStore]] = {
        VectorStoreImplementationType.FAISS.value: FAISSVectorStore
    }
    
    @classmethod
    def create(
        cls,
        store_type: Union[VectorStoreServiceType, VectorStoreImplementationType],
        settings: Settings,
        base_store: Optional[VectorStore] = None
    ) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of vector store to create (service or implementation type)
            settings: Settings instance
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If store type is unknown
        """
        # Handle service-level types
        if isinstance(store_type, VectorStoreServiceType):
            if store_type == VectorStoreServiceType.PERSISTENT:
                # For persistent store, pass factory instance and base store
                if isinstance(cls, type):
                    # If called as class method, create new instance
                    factory = cls()
                else:
                    # If called on instance, use self
                    factory = cls
                    
                return cls._service_implementations[store_type.value](
                    settings=settings,
                    factory=factory,
                    store=base_store or cls.create(settings.vector_store_impl_type, settings)
                )
            return cls._service_implementations[store_type.value](settings)
            
        # Handle implementation-level types
        if isinstance(store_type, VectorStoreImplementationType):
            # If base_store is provided and matches the requested type, use it
            if base_store and isinstance(base_store, cls._store_implementations[store_type.value]):
                return base_store
            return cls._store_implementations[store_type.value](settings)
            
        # Handle string type
        if isinstance(store_type, str):
            # Try service type first
            try:
                return cls.create(VectorStoreServiceType(store_type), settings, base_store=base_store)
            except ValueError:
                # Try implementation type
                try:
                    return cls.create(VectorStoreImplementationType(store_type), settings, base_store=base_store)
                except ValueError:
                    raise ValueError(f"Unknown store type: {store_type}")
        
        raise ValueError(f"Unknown store type: {store_type}")
    
    @classmethod
    def register_implementation(cls, name: str, implementation: Type[VectorStore]) -> None:
        """
        Register a new vector store implementation.
        
        Args:
            name: Name for the implementation
            implementation: Class implementing VectorStore
        """
        cls._implementations[name] = implementation