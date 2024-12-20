"""Tests for document processing functionality."""
import pytest
import logging

from src.core.document_processing import (
    DocumentProcessingService,
    ProcessingStrategy,
)

logger = logging.getLogger(__name__)

# Sample content for different file types
SAMPLE_HTML = """
<html>
    <body>
        <h1>Test Document</h1>
        <p>This is a test paragraph.</p>
        <p>Another paragraph with content.</p>
    </body>
</html>
"""

SAMPLE_MD = """
# Test Document

This is a test paragraph.

Another paragraph with content.
"""

def test_process_html_content(request_container):
    """Test processing HTML content."""
    service = request_container.get_document_processing_service()
    result = service.process_content(
        SAMPLE_HTML.encode(),
        file_ext='.html'
    )
    
    # BeautifulSoup should extract text without HTML tags
    assert "Test Document" in result
    assert "This is a test paragraph." in result
    assert "Another paragraph with content." in result
    assert "<html>" not in result
    assert "<body>" not in result
    assert "<p>" not in result

def test_process_markdown_content(request_container):
    """Test processing Markdown content."""
    service = request_container.get_document_processing_service()
    result = service.process_content(
        SAMPLE_MD.encode(),
        file_ext='.md'
    )
    
    # Should extract text without Markdown syntax
    assert "Test Document" in result
    assert "This is a test paragraph." in result
    assert "Another paragraph with content." in result
    assert "#" not in result

def test_process_text_content(request_container, sample_text):
    """Test processing plain text content."""
    service = request_container.get_document_processing_service()
    result = service.process_content(
        sample_text.encode(),
        file_ext='.txt'
    )
    
    assert result == sample_text

def test_process_empty_content(request_container):
    """Test that empty content raises ValueError."""
    service = request_container.get_document_processing_service()
    with pytest.raises(ValueError, match="Empty content"):
        service.process_content(b"", file_ext='.txt')

def test_process_empty_text_content(request_container):
    """Test that whitespace-only text content raises ValueError."""
    service = request_container.get_document_processing_service()
    with pytest.raises(ValueError, match="Empty text"):
        service.process_content(b"   \n  ", file_ext='.txt')

def test_process_document_with_paragraph_strategy(request_container, sample_text):
    """Test processing document with paragraph strategy."""
    service = request_container.get_document_processing_service()
    result = service.process_document(
        sample_text,
        strategy=ProcessingStrategy.PARAGRAPHS
    )
    
    assert isinstance(result, dict)

def test_process_document_with_merged_strategy(request_container, sample_text):
    """Test processing document with merged strategy."""
    service = request_container.get_document_processing_service()
    result = service.process_document(
        sample_text,
        strategy=ProcessingStrategy.MERGED
    )
    
    assert isinstance(result, dict)

def test_process_document_with_combined_strategy(request_container, sample_text):
    """Test processing document with combined strategy."""
    service = request_container.get_document_processing_service()
    result = service.process_document(
        sample_text,
        strategy=ProcessingStrategy.COMBINED
    )
    
    assert isinstance(result, dict)

def test_process_document_with_invalid_strategy(request_container, sample_text):
    """Test that invalid strategy raises ValueError."""
    service = request_container.get_document_processing_service()
    with pytest.raises(RuntimeError, match="Failed to process document: Unknown processing strategy"):
        service.process_document(
            sample_text,
            strategy="invalid_strategy"
        )

def test_process_empty_document(request_container):
    """Test that empty document raises ValueError."""
    service = request_container.get_document_processing_service()
    with pytest.raises(ValueError, match="Empty text"):
        service.process_document("")

def test_process_document_with_whitespace(request_container):
    """Test that whitespace-only document raises ValueError."""
    service = request_container.get_document_processing_service()
    with pytest.raises(ValueError, match="Empty text"):
        service.process_document("   \n  ")

def test_get_supported_types(request_container):
    """Test getting supported file types and strategies."""
    service = request_container.get_document_processing_service()
    result = service.get_supported_types()
    
    assert isinstance(result, dict)
    assert "supported_types" in result
    assert ".txt" in result["supported_types"]
    assert ".md" in result["supported_types"]
    assert ".html" in result["supported_types"]
    assert "processing_strategies" in result
    assert all(strategy.value in result["processing_strategies"] 
              for strategy in ProcessingStrategy)