"""Report the quality of the Arabic text we would index, per PDF page; it changes nothing.

Usage: python scripts/check_pdf_extraction.py data/raw/document.pdf [--show PAGE]

The numbers only flag suspicious pages. Compare flagged pages with the page image
before trusting their text. The word lists catch known damage, not every kind.
"""
import argparse
from pathlib import Path
import re
import sys

import pymupdf

from rag_hackathon.pdf_text import extract_page_texts

KNOWN_DAMAGE = re.compile(r"(?<![ء-ي])(?:[وفبكل]?)(?:األ|اإل|اآل|إال|خالل)")
DETACHED_MARK = re.compile(r"(?:^|\s)[ً-ْ]")
GLUED_NUMBER = re.compile(r"[ء-ي][0-9]|[0-9][ء-ي]")
PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
REVERSED_LINE = re.compile(r"^\s*[.\u060c\u061b\u061f:)]+\s*[\u0621-\u064a]", re.MULTILINE)
UNEXPECTED = re.compile(r"[^\u0000-ɏ؀-ۿﭐ-﷿ﹰ-﻿ -⁯]")


def page_report(number: int, text: str, images: int) -> dict:
    arabic = len(re.findall(r"[ء-ي]", text))
    return {
        "page": number,
        "chars": len(text.strip()),
        "arabic": arabic,
        "images": images,
        "presentation": len(PRESENTATION_FORMS.findall(text)),
        "unexpected": len(UNEXPECTED.findall(text)),
        "lam_alef": text.count("لا"),
        "damaged": len(KNOWN_DAMAGE.findall(text)),
        "reversed_lines": len(REVERSED_LINE.findall(text)),
        "detached_marks": len(DETACHED_MARK.findall(text)),
        "glued_numbers": len(GLUED_NUMBER.findall(text)),
    }


def flags(row: dict) -> list[str]:
    found = []
    if row["chars"] == 0:
        found.append("NO_TEXT")
    if row["arabic"] >= 200 and row["lam_alef"] == 0:
        found.append("NO_LAM_ALEF")
    for key, label in (
        ("presentation", "PRESENTATION_FORMS"),
        ("unexpected", "UNEXPECTED_CHARS"),
        ("damaged", "KNOWN_DAMAGE"),
        ("reversed_lines", "REVERSED_LINES"),
        ("detached_marks", "DETACHED_MARKS"),
        ("glued_numbers", "GLUED_NUMBERS"),
    ):
        if row[key]:
            found.append(label)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--show", type=int, help="Print the extracted text of this page.")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    columns = ["page", "chars", "arabic", "images", "presentation", "unexpected",
               "lam_alef", "damaged", "reversed_lines", "detached_marks", "glued_numbers"]
    print(" | ".join(columns) + " | flags")
    flagged = 0
    texts = extract_page_texts(args.pdf, use_ocr=False)
    with pymupdf.open(args.pdf) as pdf:
        images = [len(page.get_images()) for page in pdf]
    for number, (text, image_count) in enumerate(zip(texts, images), start=1):
        row = page_report(number, text, image_count)
        page_flags = flags(row)
        flagged += bool(page_flags)
        print(" | ".join(str(row[column]) for column in columns) + " | " + ", ".join(page_flags))
    if args.show:
        print(f"\n--- extracted text, page {args.show} ---")
        print(texts[args.show - 1])
    print(f"\nFlagged pages: {flagged} of {len(texts)}")
    print("Pages flagged NO_TEXT with images are read with OCR during indexing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
