import json
import logging
from pathlib import Path
from typing import Iterable
import unicodedata
from .pdf_text import extract_page_texts

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}
SOURCE_VERSIONS = Path("data/source_versions.json")

def load_source_versions(path: Path | None = None) -> dict[str, str]:
    """Read the version of each source, keyed by source_path; a missing file means none."""
    path = path or SOURCE_VERSIONS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def normalize_relative_path(path: Path, root: Path) -> str:
    """Return a machine-independent path: POSIX separators, Unicode NFC, case preserved."""
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())

def load_documents(folder: str | Path, ocr_files: Iterable[str] = ()) -> list[dict]:
    """Return page-level records with stable source and page metadata.

    ocr_files names PDFs, relative to folder, that must be read entirely with OCR.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {root}")

    versions = load_source_versions()
    requested_ocr = {unicodedata.normalize("NFC", Path(name).as_posix()) for name in ocr_files}
    records: list[dict] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        relative_path = normalize_relative_path(path, root)
        # Chroma rejects None values, so an undated source simply has no version field.
        version = {"version": versions[relative_path]} if relative_path in versions else {}
        if path.suffix.lower() == ".pdf":
            empty_pages: list[int] = []
            force_ocr = relative_path in requested_ocr
            requested_ocr.discard(relative_path)
            for page_index, text in enumerate(extract_page_texts(path, force_ocr=force_ocr), start=1):
                if text:
                    records.append({
                        "text": text,
                        "metadata": {
                            "source": path.name,
                            "source_path": relative_path,
                            "page": page_index,
                            "file_type": "pdf",
                            **version,
                        },
                    })
                else:
                    empty_pages.append(page_index)
            if empty_pages:
                logger.warning(
                    "%s: no extractable text on pages %s; scanned pages may need OCR.",
                    relative_path,
                    empty_pages,
                )
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                records.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "source_path": relative_path,
                        "page": 1,
                        "file_type": "txt",
                        **version,
                    },
                })
    if requested_ocr:
        raise ValueError(f"OCR was requested for PDFs that were not found: {sorted(requested_ocr)}")
    if not records:
        raise ValueError(f"No readable PDF/TXT text found in {root}. Scanned PDFs may need OCR.")
    return records
