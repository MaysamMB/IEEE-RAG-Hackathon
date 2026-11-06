"""Page text extraction for PDFs, with a repair step for damaged Arabic text.

Arabic PDFs often extract badly, and the damage depends on the extractor. On our tests
PyMuPDF kept most word spellings but reversed word order, while pdfium kept the order
but reversed some words. repair_arabic_words combines the two.
"""
import difflib
from pathlib import Path
import re
import unicodedata

import pymupdf
import pypdfium2

from .ocr import ocr_pdf_pages

ARABIC_LETTER = re.compile("[ء-ي\ufb50-\ufdff\ufe70-\ufefc]")
ARABIC_WORD = re.compile("[ء-يً-ْـ]+")
MARKS = re.compile("[ً-ْـ]")
SPACE_BEFORE_MARK = re.compile(r"\s+(?=[ً-ْ])")
IMPOSSIBLE_WORD_START = "ةى"

def _standard_letters(text: str) -> str:
    """Replace Arabic presentation forms with standard letters."""
    return unicodedata.normalize("NFKC", text)

def _plain(word: str) -> str:
    return MARKS.sub("", word)

def repair_arabic_words(ordered_text: str, reference_text: str) -> str:
    """Fix reversed words in ordered_text, using reference_text as the page vocabulary."""
    vocabulary = {
        _plain(word) for word in ARABIC_WORD.findall(SPACE_BEFORE_MARK.sub("", reference_text))
    }
    anagrams: dict[str, list[str]] = {}
    for word in vocabulary:
        anagrams.setdefault("".join(sorted(word)), []).append(word)

    def closeness(word: str, candidates: list[str]) -> float:
        return max(difflib.SequenceMatcher(None, word, c).ratio() for c in candidates)

    def repair(match: re.Match) -> str:
        word = match.group()
        plain = _plain(word)
        if not plain or plain in vocabulary:
            return word
        reversed_word = plain[::-1]
        # Tanwin keeps its final alef in place: "يلعفًا" is "فعليًا".
        reversed_before_alef = plain[-2::-1] + plain[-1]
        if reversed_before_alef in vocabulary:
            return reversed_before_alef
        if plain[0] in IMPOSSIBLE_WORD_START:
            return reversed_word
        candidates = anagrams.get("".join(sorted(plain)))
        if candidates and closeness(reversed_word, candidates) > closeness(plain, candidates):
            return reversed_word
        return word

    return ARABIC_WORD.sub(repair, ordered_text)

def _digital_page_texts(path: str | Path) -> list[str]:
    """Return the text stored in each page; pages without text stay empty."""
    with pymupdf.open(path) as pdf:
        reference_texts = [page.get_text("text").strip() for page in pdf]
    if not any(ARABIC_LETTER.search(text) for text in reference_texts):
        return reference_texts

    document = pypdfium2.PdfDocument(str(path))
    try:
        texts = []
        for index, reference in enumerate(reference_texts):
            if not ARABIC_LETTER.search(reference):
                texts.append(reference)
                continue
            ordered = document[index].get_textpage().get_text_bounded()
            repaired = repair_arabic_words(_standard_letters(ordered), _standard_letters(reference))
            texts.append(repaired.strip())
        return texts
    finally:
        document.close()

def extract_page_texts(
    path: str | Path,
    *,
    use_ocr: bool = True,
    force_ocr: bool = False,
) -> list[str]:
    """Return one text per PDF page, in page order; unreadable pages stay empty.

    Pages with an image but no text are treated as scanned and use OCR, unless use_ocr is
    False. force_ocr reads every page with OCR, for files whose stored text is too damaged.
    If OCR fails on a page, that page keeps its stored text.
    """
    texts = _digital_page_texts(path)
    if not use_ocr:
        return texts
    if force_ocr:
        pages = list(range(1, len(texts) + 1))
    else:
        with pymupdf.open(path) as pdf:
            pages = [
                number
                for number, (page, text) in enumerate(zip(pdf, texts), start=1)
                if not text and page.get_images()
            ]
    for number, text in ocr_pdf_pages(path, pages).items():
        if text:
            texts[number - 1] = text
    return texts
