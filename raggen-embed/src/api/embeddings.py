from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from core.embeddings import EmbeddingService
from core.paragraph_service import ParagraphService
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
        # Train the empty index immediately
        _vector_store.train(np.zeros((1, _vector_store.dimension), dtype=np.float32))
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
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> EmbeddingResponse:
    """Generate embedding for a single text and store it in the vector store."""
    try:
        # Configure paragraph processing if requested
        if request.paragraph_options and request.paragraph_options.enabled:
            # Create ParagraphService with custom options if provided
            paragraph_service = ParagraphService(
                min_length=request.paragraph_options.min_length or 100,
                max_length=request.paragraph_options.max_length or 1000,
                overlap=request.paragraph_options.overlap or 50
            )
            
            # Split text into paragraphs and get embeddings
            paragraphs = paragraph_service.split_text(request.text)
            paragraph_embeddings = [embedding_service.get_embedding(p) for p in paragraphs]
            
            # Merge embeddings using specified strategy
            embedding = paragraph_service.merge_embeddings(
                paragraph_embeddings,
                strategy=request.paragraph_options.merge_strategy or "mean"
            )
        else:
            # Generate single embedding without paragraph processing
            embedding = embedding_service.get_embedding(request.text)
        
        # Store embedding in vector store
        vector_id = vector_store.add_vectors(np.expand_dims(embedding, 0))[0]
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            text=request.text,
            vector_id=vector_id
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
    The embeddings are stored in the vector store for future similarity search.
    
    Requirements:
    - Each text must not be empty and must be less than 512 characters
    - Maximum 32 texts per request
    - All texts will be automatically trimmed
    
    The response includes for each text:
    - The embedding vector (384 dimensions)
    - The original text
    - The vector ID in the store
    
    Possible errors:
    - 400: Empty text list, text too long, or too many texts
    - 500: Server error during embedding generation
    """,
)
async def embed_texts(
    request: BatchTextRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
) -> BatchEmbeddingResponse:
    """Generate embeddings for multiple texts and store them in the vector store."""
    try:
        embeddings = []
        
        # Configure paragraph processing if requested
        if request.paragraph_options and request.paragraph_options.enabled:
            # Create ParagraphService with custom options if provided
            paragraph_service = ParagraphService(
                min_length=request.paragraph_options.min_length or 100,
                max_length=request.paragraph_options.max_length or 1000,
                overlap=request.paragraph_options.overlap or 50
            )
            
            # Process each text
            for text in request.texts:
                # Split text into paragraphs and get embeddings
                paragraphs = paragraph_service.split_text(text)
                paragraph_embeddings = [embedding_service.get_embedding(p) for p in paragraphs]
                
                # Merge embeddings using specified strategy
                embedding = paragraph_service.merge_embeddings(
                    paragraph_embeddings,
                    strategy=request.paragraph_options.merge_strategy or "mean"
                )
                embeddings.append(embedding)
        else:
            # Generate embeddings without paragraph processing
            embeddings = [embedding_service.get_embedding(text) for text in request.texts]
        
        # Convert list to numpy array
        embeddings_array = np.array(embeddings)
        
        # Store embeddings in vector store
        vector_ids = vector_store.add_vectors(embeddings_array)
        
        # Train the index after adding new vectors
        vector_store.train()
        
        return BatchEmbeddingResponse(
            embeddings=[
                EmbeddingResponse(
                    embedding=embedding.tolist(),
                    text=text,
                    vector_id=vector_id
                )
                for embedding, text, vector_id in zip(embeddings, request.texts, vector_ids)
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
        
        # If no results, return empty list
        if distances.size == 0 or indices.size == 0:
            return SearchResponse(
                query=request.text,
                results=[]
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