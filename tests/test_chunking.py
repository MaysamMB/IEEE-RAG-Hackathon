import pytest
from rag_hackathon.chunking import chunk_text

def test_unbroken_text_is_split_into_multiple_chunks():
    text = "A" * 1200
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) >= 3
    assert all(chunks)
    assert chunks[0].startswith("A")

def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", size=100, overlap=100)
