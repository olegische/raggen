from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'case_sensitive': False
    }
    
    # API settings
    api_title: str = "Raggen Embed API"
    api_description: str = "API for text embeddings and vector search"
    api_version: str = "1.0.0"
    
    # Model settings
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the sentence transformer model to use"
    )
    vector_dim: int = Field(
        default=384,
        description="Dimension of the embedding vectors"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to"
    )
    port: int = Field(
        default=8001,
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
        default=32,
        description="Batch size for processing multiple texts"
    )
    max_text_length: int = Field(
        default=512,
        description="Maximum length of input text"
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s",
        description="Log message format"
    )
    
    # FAISS settings
    n_clusters: int = Field(
        default=100,
        description="Number of clusters for IVF index"
    )
    n_probe: int = Field(
        default=10,
        description="Number of clusters to probe during search"
    )
    n_results: int = Field(
        default=5,
        description="Default number of results to return"
    )
    index_path: str = Field(
        default="data/faiss/index.faiss",
        description="Path to save/load FAISS index",
        env="FAISS_INDEX_PATH"
    )