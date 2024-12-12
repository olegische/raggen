import pytest
from core.text_processing import ParagraphProcessor, ParagraphConfig, Paragraph

def test_paragraph_processor_initialization():
    """Test ParagraphProcessor initialization with default config."""
    processor = ParagraphProcessor()
    assert processor.config.max_length == 1000
    assert processor.config.min_length == 100
    assert processor.config.overlap == 100
    assert processor.config.preserve_sentences is True

def test_paragraph_processor_custom_config():
    """Test ParagraphProcessor initialization with custom config."""
    config = ParagraphConfig(
        max_length=500,
        min_length=50,
        overlap=50,
        preserve_sentences=False
    )
    processor = ParagraphProcessor(config)
    assert processor.config.max_length == 500
    assert processor.config.min_length == 50
    assert processor.config.overlap == 50
    assert processor.config.preserve_sentences is False

def test_split_text_basic():
    """Test basic text splitting functionality."""
    processor = ParagraphProcessor(
        ParagraphConfig(max_length=100, min_length=10, overlap=20)
    )
    text = "This is a test text that should be split into multiple paragraphs. " * 3
    paragraphs = processor.split_text(text)
    
    assert len(paragraphs) > 1
    for p in paragraphs:
        assert isinstance(p, Paragraph)
        assert len(p.text) <= 100
        assert len(p.text) >= 10

def test_split_text_with_sentence_preservation():
    """Test text splitting with sentence preservation."""
    processor = ParagraphProcessor(
        ParagraphConfig(max_length=50, min_length=10, overlap=10, preserve_sentences=True)
    )
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    paragraphs = processor.split_text(text)
    
    for p in paragraphs:
        # Each paragraph should end with a sentence boundary
        assert p.text.strip().endswith(('.', '!', '?'))

def test_split_text_with_overlap():
    """Test text splitting with overlap."""
    processor = ParagraphProcessor(
        ParagraphConfig(max_length=50, min_length=10, overlap=20)
    )
    text = "This is a test text. " * 5
    paragraphs = processor.split_text(text)
    
    if len(paragraphs) > 1:
        # Check for overlap between consecutive paragraphs
        for i in range(len(paragraphs) - 1):
            current_end = paragraphs[i].text[-20:]  # Last 20 chars
            next_start = paragraphs[i + 1].text[:20]  # First 20 chars
            assert current_end.strip() in next_start or next_start.strip() in current_end

def test_merge_embeddings_mean():
    """Test merging embeddings with mean strategy."""
    processor = ParagraphProcessor()
    embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    merged = processor.merge_embeddings(embeddings, strategy="mean")
    assert len(merged) == 3
    assert merged[0] == 4.0  # (1 + 4 + 7) / 3
    assert merged[1] == 5.0  # (2 + 5 + 8) / 3
    assert merged[2] == 6.0  # (3 + 6 + 9) / 3

def test_merge_embeddings_weighted():
    """Test merging embeddings with weighted strategy."""
    processor = ParagraphProcessor()
    embeddings = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ]
    merged = processor.merge_embeddings(embeddings, strategy="weighted")
    # With two embeddings, weights should be approximately [0.67, 0.33]
    assert len(merged) == 3
    assert all(x > 1.0 and x < 2.0 for x in merged)

def test_empty_text():
    """Test handling of empty text."""
    processor = ParagraphProcessor()
    with pytest.raises(ValueError):
        processor.split_text("")

def test_text_shorter_than_min_length():
    """Test handling of text shorter than minimum length."""
    processor = ParagraphProcessor(
        ParagraphConfig(max_length=100, min_length=50)
    )
    text = "Short text."
    paragraphs = processor.split_text(text)
    assert len(paragraphs) == 1
    assert paragraphs[0].text == text

def test_context_preservation():
    """Test context preservation in paragraphs."""
    processor = ParagraphProcessor(
        ParagraphConfig(max_length=50, min_length=10, overlap=10)
    )
    text = "Previous context. Main content here. Following context."
    paragraphs = processor.split_text(text)
    
    for p in paragraphs:
        if p.start_pos > 0:
            assert p.context_before is not None
        if p.end_pos < len(text):
            assert p.context_after is not None

def test_merge_embeddings_invalid_strategy():
    """Test handling of invalid merge strategy."""
    processor = ParagraphProcessor()
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(ValueError):
        processor.merge_embeddings(embeddings, strategy="invalid")

def test_merge_embeddings_empty_list():
    """Test handling of empty embeddings list."""
    processor = ParagraphProcessor()
    with pytest.raises(ValueError):
        processor.merge_embeddings([], strategy="mean")