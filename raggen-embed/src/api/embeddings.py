from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from core.embeddings import EmbeddingService
from core.vector_store.faiss_store import FAISSVectorStore
from api.models import (
    TextRequest,
    BatchTextRequest,
    SearchRequest,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    SearchResponse,
    SearchResult,
    ErrorResponse,
)
from utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Global instances
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[FAISSVectorStore] = None

def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def get_vector_store() -> FAISSVectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store

@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["embeddings"],
    summary="Generate embedding for a single text",
    description="""
    Generate an embedding vector for a single text.
    
    The text must not be empty and must be less than 512 characters.
    The text will be automatically trimmed of leading and trailing whitespace.
    
    The response includes:
    - The embedding vector (384 dimensions)
    - The original text
    - The vector ID (if stored in the vector store)
    
    Possible errors:
    - 400: Text is empty or too long
    - 500: Server error during embedding generation
    """,
)
async def embed_text(
    request: TextRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """Generate embedding for a single text."""
    try:
        # Generate embedding
        embedding = embedding_service.get_embedding(request.text)
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            text=request.text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate embedding: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post(
    "/embed/batch",
    response_model=BatchEmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["embeddings"],
    summary="Generate embeddings for multiple texts",
    description="""
    Generate embedding vectors for multiple texts in a single request.
    
    Requirements:
    - Each text must not be empty and must be less than 512 characters
    - Maximum 32 texts per request
    - All texts will be automatically trimmed
    
    The response includes for each text:
    - The embedding vector (384 dimensions)
    - The original text
    - The vector ID (if stored)
    
    Possible errors:
    - 400: Empty text list, text too long, or too many texts
    - 500: Server error during embedding generation
    """,
)
async def embed_texts(
    request: BatchTextRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> BatchEmbeddingResponse:
    """Generate embeddings for multiple texts."""
    try:
        # Generate embeddings
        embeddings = embedding_service.get_embeddings(request.texts)
        
        return BatchEmbeddingResponse(
            embeddings=[
                EmbeddingResponse(
                    embedding=embedding.tolist(),
                    text=text,
                )
                for embedding, text in zip(embeddings, request.texts)
            ]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["search"],
    summary="Search for similar texts",
    description="""
    Search for texts similar to the query text using vector similarity.
    
    The process:
    1. The query text is converted to an embedding vector
    2. The vector store is searched for similar vectors
    3. Results are returned with similarity scores
    
    Requirements:
    - Query text must not be empty and must be less than 512 characters
    - Number of results (k) must be between 1 and 100
    - The vector store must be trained and contain vectors
    
    The response includes:
    - The original query text
    - A list of results, each containing:
      - The similar text
      - A similarity score (0-1, higher is more similar)
      - The vector ID in the store
    
    Possible errors:
    - 400: Invalid query text or k value
    - 500: Server error during search
    """,
)
async def search_similar(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> SearchResponse:
    """Search for similar texts."""
    try:
        # Generate query embedding
        query_embedding = embedding_service.get_embedding(request.text)
        
        # Search similar vectors
        distances, indices = vector_store.search(
            query_vectors=np.expand_dims(query_embedding, 0),
            k=request.k,
        )
        
        # Convert distances to similarity scores (1 - normalized distance)
        max_distance = np.max(distances)
        if max_distance > 0:
            scores = 1 - (distances[0] / max_distance)
        else:
            scores = np.ones_like(distances[0])
        
        # Get texts for results (in a real application, you would have a mapping of IDs to texts)
        # Here we just return placeholder texts
        results = [
            SearchResult(
                text=f"Similar text {idx}",
                score=float(score),
                vector_id=int(idx),
            )
            for idx, score in zip(indices[0], scores)
        ]
        
        return SearchResponse(
            query=request.text,
            results=results,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to search similar texts: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") 