import logging
from pathlib import Path
import unicodedata
import pymupdf

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}

def normalize_relative_path(path: Path, root: Path) -> str:
    """Return a machine-independent path: POSIX separators, Unicode NFC, case preserved."""
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())

def load_documents(folder: str | Path) -> list[dict]:
    """Return page-level records with stable source and page metadata."""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {root}")

    records: list[dict] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        relative_path = normalize_relative_path(path, root)
        if path.suffix.lower() == ".pdf":
            empty_pages: list[int] = []
            with pymupdf.open(path) as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        records.append({
                            "text": text,
                            "metadata": {
                                "source": path.name,
                                "source_path": relative_path,
                                "page": page_index,
                                "file_type": "pdf",
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
                    },
                })
    if not records:
        raise ValueError(f"No readable PDF/TXT text found in {root}. Scanned PDFs may need OCR.")
    return records
