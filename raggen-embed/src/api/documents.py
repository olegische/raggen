from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
import markdown
import logging
from typing import List

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.html'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@router.post("/documents/upload")
async def upload_document(file: UploadFile) -> JSONResponse:
    """
    Upload and process a document file.
    Supports: TXT, MD, HTML files
    """
    try:
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )

        # Process content based on file type
        processed_content = ""
        if file_ext == '.html':
            soup = BeautifulSoup(content, 'html.parser')
            processed_content = soup.get_text(separator='\n', strip=True)
        elif file_ext == '.md':
            # Convert markdown to HTML first, then extract text
            html = markdown.markdown(content.decode())
            soup = BeautifulSoup(html, 'html.parser')
            processed_content = soup.get_text(separator='\n', strip=True)
        else:  # .txt
            processed_content = content.decode()

        logger.info(f"Successfully processed document: {file.filename}")
        
        return JSONResponse(
            content={
                "message": "Document processed successfully",
                "filename": file.filename,
                "content_length": len(processed_content)
            },
            status_code=200
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
async def get_supported_types() -> JSONResponse:
    """Get list of supported document types"""
    return JSONResponse(
        content={
            "supported_types": list(SUPPORTED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE/1024/1024
        }
    )
