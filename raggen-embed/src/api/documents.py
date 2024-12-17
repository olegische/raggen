"""Document processing API endpoints."""
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from core.document_processing import (
    DocumentProcessingService,
    ProcessingStrategy
)

router = APIRouter()
logger = logging.getLogger(__name__)

# File size limits
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    strategy: ProcessingStrategy = ProcessingStrategy.PARAGRAPHS,
    doc_service: DocumentProcessingService = Depends()
) -> JSONResponse:
    """
    Upload and process a document file.
    
    The document is processed based on the selected strategy:
    - paragraphs: Split into paragraphs and save individual embeddings
    - merged: Split into paragraphs, merge embeddings, save as one
    - combined: Do both above strategies
    
    Supports: TXT, MD, HTML files
    """
    try:
        # Get supported types
        supported_types = doc_service.get_supported_types()
        
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in supported_types["supported_types"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(supported_types['supported_types'])}"
            )
    
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
    
        try:
            # Process content based on file type
            processed_content = doc_service.process_content(content, file_ext)
    
            try:
                # Process document with selected strategy
                result = doc_service.process_document(processed_content, strategy)
                
                logger.info(f"Successfully processed document: {file.filename}")
                
                return JSONResponse(
                    content={
                        "message": "Document processed successfully",
                        "filename": file.filename,
                        **result
                    },
                    status_code=200
                )
            except RuntimeError as e:
                logger.error(f"Error processing document {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document: {str(e)}"
                )
        except ValueError as e:
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
    
    except HTTPException as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing document {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/documents/supported-types")
async def get_supported_types(
    doc_service: DocumentProcessingService = Depends()
) -> JSONResponse:
    """Get list of supported document types"""
    supported_types = doc_service.get_supported_types()
    
    return JSONResponse(
        content={
            **supported_types,
            "max_file_size_mb": float(MAX_FILE_SIZE_MB)
        }
    )
