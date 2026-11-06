"""Tests for PDF text extraction and the Arabic word repair."""
import pymupdf

from rag_hackathon import pdf_text
from rag_hackathon.documents import load_documents
from rag_hackathon.pdf_text import extract_page_texts, repair_arabic_words


def test_reversed_word_is_fixed_from_reference():
    assert repair_arabic_words("فيظنت المستندات", "تنظيف المستندات") == "تنظيف المستندات"


def test_correct_word_order_is_kept():
    ordered = "تنظيف المستندات مهم"

    assert repair_arabic_words(ordered, "مهم المستندات تنظيف") == ordered


def test_reference_order_is_not_copied():
    assert repair_arabic_words("زيهجت البيانات", "البيانات تجهيز").split() == ["تجهيز", "البيانات"]


def test_word_cannot_start_with_taa_marbuta():
    assert repair_arabic_words("ةيفصو", "كلمة أخرى") == "وصفية"


def test_damaged_reference_word_guides_reversal_by_shared_letters():
    assert repair_arabic_words("زيهجت", "تجهزي") == "تجهيز"


def test_correct_word_is_kept_when_reference_has_same_letters():
    assert repair_arabic_words("مبررة", "مربرة") == "مبررة"


def test_reversed_word_with_tanwin_is_fixed():
    assert repair_arabic_words("يلعفًا", "فعل ًيا") == "فعليا"


def test_marks_on_correct_words_are_kept():
    assert repair_arabic_words("يُعَدُّ", "يعد") == "يُعَدُّ"


def test_known_word_is_kept_even_if_a_reversal_is_also_known():
    assert repair_arabic_words("سلا", "سلا لسا") == "سلا"


def test_unknown_word_is_left_unchanged():
    assert repair_arabic_words("كلمةمجهولة", "نص آخر") == "كلمةمجهولة"


def test_latin_text_and_numbers_are_unchanged():
    text = "(Chunking) Precision@k 250"

    assert repair_arabic_words(text, "نص") == text


def test_english_pdf_uses_primary_extractor_text(tmp_path):
    path = tmp_path / "english.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Plain English text")
        pdf.save(path)

    assert extract_page_texts(path) == ["Plain English text"]


def test_arabic_pdf_keeps_one_entry_per_page_including_blank_pages(tmp_path):
    path = tmp_path / "arabic.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_htmlbox(pymupdf.Rect(40, 40, 500, 200), '<p dir="rtl">تنظيف المستندات</p>')
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "English page")
        pdf.save(path)

    texts = extract_page_texts(path)

    assert len(texts) == 3
    assert texts[0]
    assert texts[1] == ""
    assert texts[2] == "English page"


def test_only_arabic_pages_go_through_the_repair(monkeypatch, tmp_path):
    path = tmp_path / "mixed.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_htmlbox(pymupdf.Rect(40, 40, 500, 200), '<p dir="rtl">تنظيف المستندات</p>')
        pdf.new_page().insert_text((72, 72), "English page")
        pdf.save(path)
    monkeypatch.setattr(pdf_text, "repair_arabic_words", lambda ordered, reference: "repaired")

    assert extract_page_texts(path) == ["repaired", "English page"]


def test_presentation_forms_become_standard_letters_before_repair(monkeypatch, tmp_path):
    path = tmp_path / "forms.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_htmlbox(pymupdf.Rect(40, 40, 500, 200), '<p dir="rtl">تنظيف</p>')
        pdf.save(path)
    seen = []
    monkeypatch.setattr(pdf_text, "repair_arabic_words", lambda ordered, reference: seen.append(reference) or ordered)

    text = extract_page_texts(path)[0]

    assert text
    assert not any(0xfb50 <= ord(char) <= 0xfefc for char in text + seen[0])


def test_loader_keeps_page_numbers_with_arabic_pdf(tmp_path):
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page().insert_htmlbox(pymupdf.Rect(40, 40, 500, 200), '<p dir="rtl">تنظيف المستندات</p>')
        pdf.save(tmp_path / "arabic.pdf")

    records = load_documents(tmp_path)

    assert [r["metadata"]["page"] for r in records] == [2]
