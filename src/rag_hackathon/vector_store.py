from collections import Counter
import logging
from pathlib import Path
import chromadb
from chromadb.errors import NotFoundError
from .config import settings

logger = logging.getLogger(__name__)

STAGING_SUFFIX = "_staging"
BACKUP_SUFFIX = "_backup"

class InterruptedRebuildError(RuntimeError):
    """The live collection is missing because an earlier rebuild did not finish."""

def _client():
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)

def _find_collection(client, name: str):
    try:
        return client.get_collection(name)
    except NotFoundError:
        return None

def _find_live_collection(client):
    """Return the live collection, or None if nothing was indexed yet."""
    live_name = settings.chroma_collection
    live = _find_collection(client, live_name)
    backup_name = live_name + BACKUP_SUFFIX
    if live is None and _find_collection(client, backup_name) is not None:
        raise InterruptedRebuildError(
            f"Collection '{live_name}' is missing but '{backup_name}' exists: an earlier "
            "rebuild did not finish. Run a full re-index with --rebuild, or restore the backup."
        )
    return live

def _delete_quietly(client, name: str) -> None:
    """Remove a leftover collection without hiding the outcome that matters more."""
    try:
        if _find_collection(client, name) is not None:
            client.delete_collection(name)
    except Exception as exc:
        logger.warning("Could not delete collection '%s': %s", name, exc)

def _index_metadata(chunk_size: int, chunk_overlap: int, documents_root: str) -> dict:
    return {
        "hnsw:space": "cosine",
        "embedding_model": settings.embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents_root": documents_root,
    }

def _validated_ids(chunks: list[dict], embeddings: list[list[float]]) -> list[str]:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")
    ids = [c["metadata"]["chunk_id"] for c in chunks]
    duplicates = sorted(chunk_id for chunk_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs: {duplicates[:5]}")
    return ids

def _write(
    collection,
    ids: list[str],
    chunks: list[dict],
    embeddings: list[list[float]],
    batch_size: int,
) -> None:
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        collection.upsert(
            ids=ids[start:stop],
            documents=[c["text"] for c in chunks[start:stop]],
            metadatas=[c["metadata"] for c in chunks[start:stop]],
            embeddings=embeddings[start:stop],
        )

def get_collection():
    """Return the live collection, or None if nothing is indexed. Never creates or deletes."""
    return _find_live_collection(_client())

def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    documents_root: str,
) -> int:
    """Add or update chunks without deleting; refuse to mix incompatible index settings.

    Chunk IDs are relative to documents_root, so one collection accepts one root only:
    a second root could silently overwrite chunks that share a relative path.
    """
    ids = _validated_ids(chunks, embeddings)
    if not chunks:
        return 0
    client = _client()
    expected = _index_metadata(chunk_size, chunk_overlap, documents_root)
    collection = _find_live_collection(client)
    if collection is None:
        collection = client.create_collection(settings.chroma_collection, metadata=expected)
    else:
        stored = collection.metadata or {}
        differing = sorted(key for key in expected if stored.get(key) != expected[key])
        if differing:
            raise ValueError(
                f"Collection '{settings.chroma_collection}' was indexed with different "
                f"settings ({', '.join(differing)}). Run a full re-index with --rebuild."
            )
    _write(collection, ids, chunks, embeddings, client.get_max_batch_size())
    return len(chunks)

def rebuild_collection(
    chunks: list[dict],
    embeddings: list[list[float]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    documents_root: str,
) -> int:
    """Replace the live collection; this is the only operation that removes indexed chunks.

    The replacement is fully written to a staging collection before the swap, and the old
    collection is kept as a backup until the swap succeeds. It also repairs an interrupted
    rebuild, because it needs no live collection.
    """
    ids = _validated_ids(chunks, embeddings)
    if not chunks:
        raise ValueError("Refusing to replace the collection with an empty index")

    client = _client()
    live_name = settings.chroma_collection
    staging_name = live_name + STAGING_SUFFIX
    backup_name = live_name + BACKUP_SUFFIX

    if _find_collection(client, staging_name) is not None:
        client.delete_collection(staging_name)
    staging = client.create_collection(
        staging_name, metadata=_index_metadata(chunk_size, chunk_overlap, documents_root)
    )
    try:
        _write(staging, ids, chunks, embeddings, client.get_max_batch_size())
        if staging.count() != len(chunks):
            raise RuntimeError("Staging collection is incomplete; live index was not changed")
    except Exception:
        _delete_quietly(client, staging_name)
        raise

    live = _find_collection(client, live_name)
    live_moved = False
    try:
        if live is not None:
            if _find_collection(client, backup_name) is not None:
                client.delete_collection(backup_name)
            live.modify(name=backup_name)
            live_moved = True
        staging.modify(name=live_name)
    except Exception:
        # If this rollback fails, staging survives as a second complete copy of the data.
        if live_moved:
            live.modify(name=live_name)
        _delete_quietly(client, staging_name)
        raise

    _delete_quietly(client, backup_name)
    return len(chunks)
