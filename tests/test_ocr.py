"""OCR routing and cache tests; the recognizer is faked, so no model is needed."""
from dataclasses import replace
import logging

import pymupdf
import pytest

from rag_hackathon import config, ocr
from rag_hackathon.documents import load_documents
from rag_hackathon.pdf_text import extract_page_texts


@pytest.fixture
def recognized(monkeypatch, tmp_path):
    """Replace OCR with a fake that records the pages it was asked to read."""
    pages = []

    def fake(page):
        pages.append(page.number + 1)
        return f"ocr text {page.number + 1}"

    monkeypatch.setattr(ocr, "recognize_page", fake)
    monkeypatch.setattr(ocr, "settings", replace(config.settings, ocr_cache_dir=str(tmp_path / "cache")))
    return pages


def add_scanned_page(pdf):
    page = pdf.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    pixmap.clear_with(255)
    page.insert_image(page.rect, pixmap=pixmap)


def make_pdf(path):
    """Page 1 has text, page 2 is scanned, and page 3 is blank."""
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Text page")
        add_scanned_page(pdf)
        pdf.new_page()
        pdf.save(path)
    return path


def test_only_scanned_pages_use_ocr(recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")

    assert extract_page_texts(path) == ["Text page", "ocr text 2", ""]
    assert recognized == [2]


def test_loader_keeps_page_number_of_scanned_page(recognized, tmp_path, caplog):
    make_pdf(tmp_path / "mixed.pdf")

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.documents"):
        records = load_documents(tmp_path)

    assert [(r["metadata"]["page"], r["text"]) for r in records] == [(1, "Text page"), (2, "ocr text 2")]
    assert "[3]" in caplog.text


def test_requested_file_uses_ocr_on_every_page(recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")

    assert extract_page_texts(path, force_ocr=True) == ["ocr text 1", "ocr text 2", "ocr text 3"]


def test_ocr_can_be_disabled(recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")

    assert extract_page_texts(path, use_ocr=False) == ["Text page", "", ""]
    assert recognized == []


def test_loader_forces_ocr_only_for_named_file(recognized, tmp_path):
    make_pdf(tmp_path / "wanted.pdf")
    make_pdf(tmp_path / "other.pdf")

    records = load_documents(tmp_path, ocr_files=["wanted.pdf"])

    texts = {(r["metadata"]["source"], r["metadata"]["page"]): r["text"] for r in records}
    assert texts["wanted.pdf", 1] == "ocr text 1"
    assert texts["other.pdf", 1] == "Text page"


def test_unknown_ocr_file_is_rejected(recognized, tmp_path):
    make_pdf(tmp_path / "real.pdf")

    with pytest.raises(ValueError, match="missing.pdf"):
        load_documents(tmp_path, ocr_files=["missing.pdf"])


def test_results_are_reused_from_cache(recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")

    extract_page_texts(path)
    assert extract_page_texts(path)[1] == "ocr text 2"

    assert recognized == [2]


def test_changed_file_is_read_again(recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")
    extract_page_texts(path)
    with pymupdf.open() as pdf:
        add_scanned_page(pdf)
        add_scanned_page(pdf)
        pdf.save(path)

    extract_page_texts(path)

    assert recognized == [2, 1, 2]


def test_interrupted_ocr_keeps_finished_pages_and_resumes(monkeypatch, recognized, tmp_path, caplog):
    path = make_pdf(tmp_path / "mixed.pdf")
    working = ocr.recognize_page

    def fail_on_second_page(page):
        if page.number == 1:
            raise RuntimeError("no network")
        return working(page)

    monkeypatch.setattr(ocr, "recognize_page", fail_on_second_page)
    texts = extract_page_texts(path, force_ocr=True)

    assert texts == ["ocr text 1", "", ""]
    assert "no network" in caplog.text

    monkeypatch.setattr(ocr, "recognize_page", working)
    assert extract_page_texts(path, force_ocr=True) == ["ocr text 1", "ocr text 2", "ocr text 3"]
    assert recognized == [1, 2, 3]


def test_failed_forced_ocr_keeps_digital_text(monkeypatch, recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")

    def fail(_):
        raise RuntimeError("no model")

    monkeypatch.setattr(ocr, "recognize_page", fail)

    assert extract_page_texts(path, force_ocr=True) == ["Text page", "", ""]


def test_empty_ocr_result_keeps_digital_text(monkeypatch, recognized, tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf")
    monkeypatch.setattr(ocr, "recognize_page", lambda page: "")

    assert extract_page_texts(path, force_ocr=True)[0] == "Text page"


def test_no_pages_means_no_ocr_work(recognized):
    assert ocr.ocr_pdf_pages("unused.pdf", []) == {}
