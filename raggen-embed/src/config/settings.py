from typing import List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum


class IndexType(str, Enum):
    """Available FAISS index types."""
    FLAT_L2 = "flat_l2"  # Exact search, no training needed
    IVF_FLAT = "ivf_flat"  # Approximate search with clustering, requires training
    IVF_PQ = "ivf_pq"  # Compressed vectors with clustering, requires training
    HNSW_FLAT = "hnsw_flat"  # Graph-based search, no training but has parameters


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'case_sensitive': False
    }
    
    # API settings
    api_title: str = Field(
        default="Raggen Embed API",
        description="API title",
        env="API_TITLE"
    )
    api_description: str = Field(
        default="API for text embeddings and vector search",
        description="API description",
        env="API_DESCRIPTION"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version",
        env="API_VERSION"
    )
    
    # Model settings
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the sentence transformer model to use",
        env="MODEL_NAME"
    )
    vector_dim: int = Field(
        default=384,
        description="Dimension of the embedding vectors",
        env="VECTOR_DIM"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to",
        env="HOST"
    )
    port: int = Field(
        default=8001,
        description="Port to bind the server to",
        env="PORT"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins",
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
        env="CORS_ALLOW_CREDENTIALS"
    )
    cors_allow_methods: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP methods",
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="List of allowed HTTP headers",
        env="CORS_ALLOW_HEADERS"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=32,
        description="Batch size for processing multiple texts",
        env="BATCH_SIZE"
    )
    max_text_length: int = Field(
        default=512,
        description="Maximum length of input text",
        env="MAX_TEXT_LENGTH"
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        env="REQUEST_TIMEOUT"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        env="LOG_LEVEL"
    )
    log_format: str = Field(
        default="%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s",
        description="Log message format",
        env="LOG_FORMAT"
    )
    
    # FAISS settings
    faiss_index_type: IndexType = Field(
        default=IndexType.FLAT_L2,  # Changed default to FLAT_L2 for exact search
        description="Type of FAISS index to use. Options:\n"
                   "- flat_l2: Exact search, no training needed (best for accurate paragraph search)\n"
                   "- ivf_flat: Approximate search with clustering, requires training (faster but less accurate)\n"
                   "- ivf_pq: Compressed vectors with clustering, requires training (memory efficient)\n"
                   "- hnsw_flat: Graph-based search, no training but has parameters (good balance)",
        env="FAISS_INDEX_TYPE"
    )
    
    # IVF settings (used by ivf_flat and ivf_pq)
    n_clusters: int = Field(
        default=100,
        description="Number of clusters for IVF index (used by ivf_flat and ivf_pq)",
        env="N_CLUSTERS"
    )
    n_probe: int = Field(
        default=10,
        description="Number of clusters to probe during search (used by ivf_flat and ivf_pq)",
        env="N_PROBE"
    )
    
    # PQ settings (used by ivf_pq)
    pq_m: int = Field(
        default=8,
        description="Number of sub-vectors in Product Quantization (used by ivf_pq). Must divide vector_dim evenly.",
        env="PQ_M"
    )
    pq_bits: int = Field(
        default=8,
        description="Number of bits per sub-vector in Product Quantization (used by ivf_pq). Must be 8, 12, or 16.",
        env="PQ_BITS"
    )
    
    # HNSW settings (used by hnsw_flat)
    hnsw_m: int = Field(
        default=16,
        description="Number of neighbors for HNSW graph (used by hnsw_flat)",
        env="HNSW_M"
    )
    hnsw_ef_construction: int = Field(
        default=40,
        description="Exploration factor during HNSW construction (used by hnsw_flat)",
        env="HNSW_EF_CONSTRUCTION"
    )
    hnsw_ef_search: int = Field(
        default=16,
        description="Exploration factor during HNSW search (used by hnsw_flat)",
        env="HNSW_EF_SEARCH"
    )
    
    # General search settings
    n_results: int = Field(
        default=5,
        description="Default number of results to return",
        env="N_RESULTS"
    )
    faiss_index_path: str = Field(
        default="/app/data/faiss/index.faiss",
        description="Path to save/load FAISS index",
        env="FAISS_INDEX_PATH"
    )
    
    # Paragraph processing settings
    paragraph_max_length: int = Field(
        default=1000,
        description="Maximum length of a paragraph",
        env="PARAGRAPH_MAX_LENGTH"
    )
    paragraph_min_length: int = Field(
        default=100,
        description="Minimum length of a paragraph",
        env="PARAGRAPH_MIN_LENGTH"
    )
    paragraph_overlap: int = Field(
        default=100,
        description="Number of characters to overlap between paragraphs",
        env="PARAGRAPH_OVERLAP"
    )
    preserve_sentences: bool = Field(
        default=True,
        description="Whether to preserve sentence boundaries when splitting paragraphs",
        env="PRESERVE_SENTENCES"
    )
    context_window_size: int = Field(
        default=200,
        description="Size of context window before and after paragraph",
        env="CONTEXT_WINDOW_SIZE"
    )
    embedding_merge_strategy: str = Field(
        default="mean",
        description="Strategy for merging paragraph embeddings (mean or weighted)",
        env="EMBEDDING_MERGE_STRATEGY"
    )
    
    @property
    def requires_training(self) -> bool:
        """Check if the current index type requires training."""
        return self.faiss_index_type in {IndexType.IVF_FLAT, IndexType.IVF_PQ}
