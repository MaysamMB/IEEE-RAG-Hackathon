from pathlib import Path
import fitz

SUPPORTED_SUFFIXES = {".pdf", ".txt"}

def load_documents(folder: str | Path) -> list[dict]:
    """Return page-level records with stable source and page metadata."""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {root}")

    records: list[dict] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        records.append({
                            "text": text,
                            "metadata": {
                                "source": path.name,
                                "source_path": str(path),
                                "page": page_index,
                                "file_type": "pdf",
                            },
                        })
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                records.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "source_path": str(path),
                        "page": 1,
                        "file_type": "txt",
                    },
                })
    if not records:
        raise ValueError(f"No readable PDF/TXT text found in {root}. Scanned PDFs may need OCR.")
    return records
