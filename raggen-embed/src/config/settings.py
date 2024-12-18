"""Application settings."""
from typing import List, Literal
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)

class IndexType(str, Enum):
    """Available FAISS index types."""
    FLAT_L2 = "flat_l2"  # Exact search, no training needed
    IVF_FLAT = "ivf_flat"  # Approximate search with clustering, requires training
    IVF_PQ = "ivf_pq"  # Compressed vectors with clustering, requires training
    HNSW_FLAT = "hnsw_flat"  # Graph-based search, no training but has parameters

class TextSplitStrategy(str, Enum):
    """Available text splitting strategies."""
    SLIDING_WINDOW = "sliding_window"
    PARAGRAPH = "paragraph"

class VectorStoreServiceType(str, Enum):
    """High-level vector store types for service layer."""
    PERSISTENT = "persistent"  # FAISS with persistence

class VectorStoreImplementationType(str, Enum):
    """Low-level vector store implementations."""
    FAISS = "faiss"  # Only FAISS for now, can add more implementations later

class Settings(BaseSettings):
    """Application settings."""
    
    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'case_sensitive': False
    }
    
    # API settings
    api_title: str = Field(
        default=os.getenv("API_TITLE", "Raggen Embed API"),
        description="API title"
    )
    api_description: str = Field(
        default=os.getenv("API_DESCRIPTION", "API for text embeddings and vector search"),
        description="API description"
    )
    api_version: str = Field(
        default=os.getenv("API_VERSION", "1.0.0"),
        description="API version"
    )
    
    # Model settings
    model_name: str = Field(
        default=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        description="Name of the sentence transformer model to use"
    )
    vector_dim: int = Field(
        default=int(os.getenv("VECTOR_DIM", "384")),
        description="Dimension of the embedding vectors"
    )
    
    # Server settings
    host: str = Field(
        default=os.getenv("HOST", "0.0.0.0"),
        description="Host to bind the server to"
    )
    port: int = Field(
        default=int(os.getenv("PORT", "8001")),
        description="Port to bind the server to"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP methods"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP headers"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=int(os.getenv("BATCH_SIZE", "32")),
        description="Batch size for processing multiple texts"
    )
    max_text_length: int = Field(
        default=int(os.getenv("MAX_TEXT_LENGTH", "512")),
        description="Maximum length of input text"
    )
    request_timeout: int = Field(
        default=int(os.getenv("REQUEST_TIMEOUT", "30")),
        description="Request timeout in seconds"
    )
    tokenizers_parallelism: bool = Field(
        default=bool(os.getenv("TOKENIZERS_PARALLELISM", "false")),
        description="Whether to enable parallelism in tokenizers. Set to false to avoid fork-related issues."
    )
    omp_num_threads: int = Field(
        default=int(os.getenv("OMP_NUM_THREADS", "1")),
        description="Number of OpenMP threads. Set to 1 to avoid parallelism issues."
    )
    mkl_num_threads: int = Field(
        default=int(os.getenv("MKL_NUM_THREADS", "1")),
        description="Number of MKL threads. Set to 1 to avoid parallelism issues."
    )
    
    # Logging settings
    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level"
    )
    log_format: str = Field(
        default=os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s"),
        description="Log message format"
    )
    
    # FAISS settings
    faiss_index_type: IndexType = Field(
        default=IndexType(os.getenv("FAISS_INDEX_TYPE", "flat_l2")),
        description="Type of FAISS index to use. Options:\n"
                   "- flat_l2: Exact search, no training needed (best for accurate paragraph search)\n"
                   "- ivf_flat: Approximate search with clustering, requires training (faster but less accurate)\n"
                   "- ivf_pq: Compressed vectors with clustering, requires training (memory efficient)\n"
                   "- hnsw_flat: Graph-based search, no training but has parameters (good balance)"
    )
    
    # IVF settings (used by ivf_flat and ivf_pq)
    n_clusters: int = Field(
        default=int(os.getenv("N_CLUSTERS", "100")),
        description="Number of clusters for IVF index (used by ivf_flat and ivf_pq)"
    )
    n_probe: int = Field(
        default=int(os.getenv("N_PROBE", "10")),
        description="Number of clusters to probe during search (used by ivf_flat and ivf_pq)"
    )
    
    # PQ settings (used by ivf_pq)
    pq_m: int = Field(
        default=int(os.getenv("PQ_M", "8")),
        description="Number of sub-vectors in Product Quantization (used by ivf_pq). Must divide vector_dim evenly."
    )
    pq_bits: int = Field(
        default=int(os.getenv("PQ_BITS", "8")),
        description="Number of bits per sub-vector in Product Quantization (used by ivf_pq). Must be 8, 12, or 16."
    )
    
    # HNSW settings (used by hnsw_flat)
    hnsw_m: int = Field(
        default=int(os.getenv("HNSW_M", "16")),
        description="Number of neighbors for HNSW graph (used by hnsw_flat)"
    )
    hnsw_ef_construction: int = Field(
        default=int(os.getenv("HNSW_EF_CONSTRUCTION", "40")),
        description="Exploration factor during HNSW construction (used by hnsw_flat)"
    )
    hnsw_ef_search: int = Field(
        default=int(os.getenv("HNSW_EF_SEARCH", "16")),
        description="Exploration factor during HNSW search (used by hnsw_flat)"
    )
    
    # Vector store settings
    vector_store_service_type: VectorStoreServiceType = Field(
        default=VectorStoreServiceType(os.getenv("VECTOR_STORE_TYPE", "persistent")),
        description="High-level vector store type (faiss for direct FAISS usage or persistent for FAISS with persistence)"
    )
    
    vector_store_impl_type: VectorStoreImplementationType = Field(
        default=VectorStoreImplementationType.FAISS,
        description="Low-level vector store implementation type"
    )
    
    # General search settings
    n_results: int = Field(
        default=int(os.getenv("N_RESULTS", "5")),
        description="Default number of results to return"
    )
    faiss_index_path: str = Field(
        default=os.getenv("FAISS_INDEX_PATH", "/app/data/faiss/index.faiss"),
        description="Path to save/load FAISS index"
    )
    
    # Text splitting settings
    text_split_strategy: TextSplitStrategy = Field(
        default=TextSplitStrategy(os.getenv("TEXT_SPLIT_STRATEGY", "sliding_window")),
        description="Strategy for splitting text (sliding_window or paragraph)"
    )
    text_min_length: int = Field(
        default=int(os.getenv("TEXT_MIN_LENGTH", "100")),
        description="Minimum length of a text chunk"
    )
    text_max_length: int = Field(
        default=int(os.getenv("TEXT_MAX_LENGTH", "1000")),
        description="Maximum length of a text chunk"
    )
    text_overlap: int = Field(
        default=int(os.getenv("TEXT_OVERLAP", "50")),
        description="Number of characters to overlap between chunks (for sliding window)"
    )
    preserve_sentences: bool = Field(
        default=bool(os.getenv("PRESERVE_SENTENCES", "True")),
        description="Whether to preserve sentence boundaries when splitting text"
    )
    context_window_size: int = Field(
        default=int(os.getenv("CONTEXT_WINDOW_SIZE", "200")),
        description="Size of context window before and after chunk"
    )
    embedding_merge_strategy: str = Field(
        default=os.getenv("EMBEDDING_MERGE_STRATEGY", "mean"),
        description="Strategy for merging embeddings (mean or weighted)"
    )
    
    @property
    def requires_training(self) -> bool:
        """Check if the current index type requires training."""
        return self.faiss_index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}

# Global settings instance with caching
_settings_instance = None

def get_settings() -> Settings:
    """Get application settings with caching."""
    global _settings_instance
    if _settings_instance is None:
        logger.info("[Settings] Creating new Settings instance")
        _settings_instance = Settings()
    return _settings_instance

def reset_settings():
    """Reset settings instance. Use only in tests."""
    global _settings_instance
    logger.info("[Settings] Resetting Settings instance")
    _settings_instance = None
