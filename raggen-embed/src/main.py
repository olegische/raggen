from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.openapi.utils import get_openapi
import time
import uuid

from config.settings import Settings
from api.embeddings import router as embeddings_router
from api.documents import router as documents_router
from utils.logging import get_logger

logger = get_logger(__name__)

def create_app(settings: Settings = None) -> FastAPI:
    """Create FastAPI application."""
    if settings is None:
        settings = Settings()
    
    app = FastAPI(
        title=settings.api_title,
        description="""
        # Text Embedding and Document Processing API

        This API provides endpoints for text vectorization, document processing, and similarity search.
        
        ## Features
        
        - Generate embeddings for single texts
        - Generate embeddings for multiple texts in batch
        - Search for similar texts using vector similarity
        - Process and analyze documents (TXT, MD, HTML)
        
        ## Authentication
        
        Currently, the API does not require authentication.
        
        ## Rate Limiting
        
        - Single text embedding: No limit
        - Batch text embedding: Maximum 32 texts per request
        - Similarity search: Maximum 100 results per query
        - Document upload: Maximum 10MB per file
        
        ## Text Limits
        
        - Maximum text length: 512 characters
        - Texts must not be empty
        - Texts are automatically trimmed
        
        ## Vector Store
        
        The API uses FAISS for efficient similarity search with the following configuration:
        - Index type: IVF Flat
        - Number of clusters: Configurable
        - Number of probes: Configurable
        
        ## Document Processing
        
        Supported document types:
        - Plain text (.txt)
        - Markdown (.md)
        - HTML (.html)
        """,
        version=settings.api_version,
        docs_url=None,  # Disable default docs
        redoc_url=None,  # Disable default redoc
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add request ID to each request."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        logger.info("Incoming request: %s %s", request.method, request.url.path)
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed: %s %s - Status: %d - Time: %.2f seconds",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )
        
        return response
    
    # Include routers
    app.include_router(embeddings_router, prefix="/api/v1", tags=["embeddings"])
    app.include_router(documents_router, prefix="/api/v1", tags=["documents"])
    
    def custom_openapi():
        """Generate custom OpenAPI schema."""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add security schemes if needed
        # openapi_schema["components"]["securitySchemes"] = {...}
        
        # Add examples
        openapi_schema["components"]["examples"] = {
            "single_text": {
                "summary": "Single text embedding request",
                "value": {
                    "text": "This is a sample text to generate embedding for"
                }
            },
            "batch_texts": {
                "summary": "Batch text embedding request",
                "value": {
                    "texts": [
                        "First sample text",
                        "Second sample text",
                        "Third sample text"
                    ]
                }
            },
            "search_query": {
                "summary": "Similarity search request",
                "value": {
                    "text": "Sample query text",
                    "k": 5
                }
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        )
    
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        """ReDoc UI."""
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        logger.debug("Health check requested")
        return {"status": "ok"}
    
    return app

# Create default application instance
app = create_app()
