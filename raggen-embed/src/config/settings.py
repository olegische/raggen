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
    n_clusters: int = Field(
        default=100,
        description="Number of clusters for IVF index",
        env="N_CLUSTERS"
    )
    n_probe: int = Field(
        default=10,
        description="Number of clusters to probe during search",
        env="N_PROBE"
    )
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
