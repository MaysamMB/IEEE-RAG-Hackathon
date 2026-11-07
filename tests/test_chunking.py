import pytest
from rag_hackathon.chunking import article_labels, chunk_records, chunk_text

def test_unbroken_text_is_split_into_multiple_chunks():
    text = "A" * 1200
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) >= 3
    assert all(chunks)
    assert chunks[0].startswith("A")

def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", size=100, overlap=100)


def article_record(text, path="bylaw.txt", page=1):
    return {"text": text, "metadata": {"source": path, "source_path": path, "page": page}}


def articles_of(records, size=60, overlap=0):
    return [chunk["metadata"].get("article") for chunk in chunk_records(records, size, overlap)]


def test_chunk_records_the_article_it_starts_with():
    assert articles_of([article_record("المادة (3): الحد الأعلى")]) == ["3"]


def test_plain_and_spaced_article_headings_are_recognized():
    assert article_labels("المادة 3: نص", None) == ["3"]
    assert article_labels("المادة ( 12 ): نص", None) == ["12"]


def test_mention_of_a_study_subject_is_not_an_article():
    assert article_labels("المادة التعليمية لها 3 ساعات", None) == []
    assert article_labels("نص يذكر المادة (5): داخل السطر", None) == []


def test_chunk_covering_two_articles_lists_both():
    assert article_labels("المادة (1): أ\nالمادة (2): ب", None) == ["1", "2"]


def test_continued_text_keeps_the_previous_article():
    assert article_labels("بقية النص\nالمادة (4): ج", "3") == ["3", "4"]


def test_article_continues_across_chunks_and_pages():
    first = article_record("المادة (7): " + "نص " * 10, page=1)
    second = article_record("تكملة المادة على الصفحة التالية", page=2)

    assert articles_of([first, second]) == ["7", "7"]


def test_article_does_not_carry_into_another_file():
    records = [article_record("المادة (7): نص"), article_record("مقدمة عامة", path="other.txt")]

    assert articles_of(records) == ["7", None]


def test_chunk_without_articles_has_no_article_metadata():
    chunk = chunk_records([article_record("نص عادي")], 60, 0)[0]

    assert "article" not in chunk["metadata"]
