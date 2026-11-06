from pathlib import Path
from .config import settings
from .documents import load_documents
from .chunking import chunk_records
from .embeddings import embed_texts
from .vector_store import rebuild_collection, upsert_chunks

def ingest(folder: str = "data/raw", *, rebuild: bool = False) -> int:
    """Index documents. Only rebuild=True replaces existing chunks; all reads and
    embedding work happen first, so a failure there leaves the live index untouched."""
    pages = load_documents(folder)
    chunks = chunk_records(pages, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts([c["text"] for c in chunks])
    write = rebuild_collection if rebuild else upsert_chunks
    count = write(
        chunks,
        vectors,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        documents_root=Path(folder).resolve().as_posix(),
    )
    action = "Rebuilt index with" if rebuild else "Indexed"
    print(f"{action} {count} chunks from {len(pages)} readable page/document records.")
    return count

if __name__ == "__main__":
    ingest()
