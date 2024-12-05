from typing import List, Optional
from pydantic import BaseModel, Field, validator

class TextRequest(BaseModel):
    """Request model for text embedding."""
    text: str = Field(
        ...,
        description="Text to generate embedding for. Must not be empty and must be less than 512 characters.",
        example="This is a sample text to generate embedding for",
        min_length=1,
        max_length=512,
    )
    
    @validator("text")
    def text_not_empty(cls, v):
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 512:
            raise ValueError("Text cannot be longer than 512 characters")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text to generate embedding for"
            }
        }

class BatchTextRequest(BaseModel):
    """Request model for batch text embedding."""
    texts: List[str] = Field(
        ...,
        description="List of texts to generate embeddings for. Each text must not be empty and must be less than 512 characters. Maximum 32 texts per request.",
        max_length=32,
        example=[
            "First sample text",
            "Second sample text",
            "Third sample text"
        ]
    )
    
    @validator("texts")
    def texts_not_empty(cls, v):
        """Validate texts are not empty."""
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at position {i} cannot be empty")
            if len(text) > 512:
                raise ValueError(f"Text at position {i} cannot be longer than 512 characters")
        return [text.strip() for text in v]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "First sample text",
                    "Second sample text",
                    "Third sample text"
                ]
            }
        }

class SearchRequest(BaseModel):
    """Request model for similarity search."""
    text: str = Field(
        ...,
        description="Query text to search similar texts for. Must not be empty and must be less than 512 characters.",
        example="Sample query text",
        min_length=1,
        max_length=512,
    )
    k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of similar texts to return (1-100, default: 5)",
        example=5
    )
    
    @validator("text")
    def text_not_empty(cls, v):
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 512:
            raise ValueError("Text cannot be longer than 512 characters")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Sample query text",
                "k": 5
            }
        }

class EmbeddingResponse(BaseModel):
    """Response model for text embedding."""
    embedding: List[float] = Field(
        ...,
        description="Embedding vector (list of 384 floating point numbers)",
        example=[0.1, 0.2, 0.3, 0.4, 0.5]  # Shortened for readability
    )
    text: str = Field(
        ...,
        description="Original text that was embedded",
        example="This is a sample text"
    )
    vector_id: int = Field(
        ...,
        description="ID of the vector in the store",
        example=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "text": "This is a sample text",
                "vector_id": 1
            }
        }

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch text embedding."""
    embeddings: List[EmbeddingResponse] = Field(
        ...,
        description="List of embeddings for each input text"
    )

class SearchResult(BaseModel):
    """Model for a single search result."""
    text: str = Field(
        ...,
        description="Similar text that was found",
        example="This is a similar text"
    )
    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Similarity score (0-1, higher is more similar)",
        example=0.85
    )
    vector_id: int = Field(
        ...,
        description="ID of the vector in the store",
        example=1
    )

class SearchResponse(BaseModel):
    """Response model for similarity search."""
    query: str = Field(
        ...,
        description="Original query text",
        example="Sample query text"
    )
    results: List[SearchResult] = Field(
        ...,
        description="List of similar texts with their similarity scores"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Sample query text",
                "results": [
                    {
                        "text": "First similar text",
                        "score": 0.85,
                        "vector_id": 1
                    },
                    {
                        "text": "Second similar text",
                        "score": 0.75,
                        "vector_id": 2
                    }
                ]
            }
        }

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(
        ...,
        description="Error message",
        example="Text cannot be empty"
    )
    details: Optional[str] = Field(
        None,
        description="Additional error details",
        example="The provided text was empty or contained only whitespace"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Text cannot be empty",
                "details": "The provided text was empty or contained only whitespace"
            }
        } 