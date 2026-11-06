"""OCR for scanned or badly extracted PDF pages, with results cached per file.

OCR is slow on a CPU, so results are stored by file content and each page is saved as
soon as it finishes. easyocr is imported only when a page really needs it.
"""
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pymupdf

from .config import settings

logger = logging.getLogger(__name__)

OCR_DPI = 200
OCR_LANGUAGES = ("ar", "en")
CACHE_FORMAT = f"easyocr-{OCR_DPI}dpi-{'-'.join(OCR_LANGUAGES)}"

@lru_cache(maxsize=1)
def _get_reader():
    import easyocr

    return easyocr.Reader(list(OCR_LANGUAGES), gpu=False, verbose=False)

def recognize_page(page) -> str:
    pixmap = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csRGB)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    return "\n".join(_get_reader().readtext(image, detail=0, paragraph=True)).strip()

def _cache_path(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Path(settings.ocr_cache_dir) / f"{digest}-{CACHE_FORMAT}.json"

def _load_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))

def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    temporary.replace(cache_path)

def ocr_pdf_pages(path: str | Path, page_numbers: Iterable[int]) -> dict[int, str]:
    """Return OCR text for 1-based pages; pages that could not be read are left out."""
    path = Path(path)
    wanted = list(page_numbers)
    if not wanted:
        return {}
    cache_path = _cache_path(path)
    cache = _load_cache(cache_path)
    missing = [number for number in wanted if str(number) not in cache]
    if missing:
        logger.info("Running OCR on %d pages of %s; this can take minutes.", len(missing), path.name)
        with pymupdf.open(path) as pdf:
            for number in missing:
                try:
                    cache[str(number)] = recognize_page(pdf[number - 1])
                except Exception as exc:
                    logger.warning("OCR stopped at %s page %d: %s", path.name, number, exc)
                    break
                _save_cache(cache_path, cache)
    return {number: cache[str(number)] for number in wanted if str(number) in cache}
