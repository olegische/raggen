from config.settings import Settings
from .faiss_store import FAISSVectorStore
from .persistent_store import PersistentStore

# Example 1: Using default settings
store = PersistentStore()  # Uses settings.py defaults

# Example 2: Custom dimension but default settings otherwise
custom_store = FAISSVectorStore(dimension=512)  # Override dimension
persistent_store = PersistentStore(store=custom_store)  # Inject custom store

# Example 3: Custom path
store_with_path = PersistentStore(
    index_path="/custom/path/index.faiss"  # Override default path
)

# Example 4: Full customization
settings = Settings()  # Get settings instance
custom_faiss = FAISSVectorStore(
    dimension=settings.vector_dim  # Use setting but explicit
)
custom_persistent = PersistentStore(
    store=custom_faiss,
    index_path=settings.faiss_index_path,  # Use setting but explicit
    auto_save=True
)

# Example usage:
def process_vectors(vectors, store: PersistentStore = None):
    """
    Process vectors using a store with dependency injection.
    
    Args:
        vectors: Vectors to process
        store: Store to use (default: create new with settings)
    """
    # If no store provided, create with defaults
    store = store or PersistentStore()
    
    # Use the store
    store.add(vectors)
    results = store.search(vectors)
    return results