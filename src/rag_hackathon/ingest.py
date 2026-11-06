from .config import settings
from .documents import load_documents
from .chunking import chunk_records
from .embeddings import embed_texts
from .vector_store import upsert_chunks

def ingest(folder: str = "data/raw") -> int:
    pages = load_documents(folder)
    chunks = chunk_records(pages, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts([c["text"] for c in chunks])
    count = upsert_chunks(chunks, vectors)
    print(f"Indexed {count} chunks from {len(pages)} readable page/document records.")
    return count

if __name__ == "__main__":
    ingest()
