from .config import settings
from .embeddings import embed_texts
from .vector_store import get_collection

def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    if not question.strip():
        return []
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return []
    query_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k or settings.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if settings.max_cosine_distance is not None and distance > settings.max_cosine_distance:
            continue
        hits.append({"text": document, "metadata": metadata, "distance": float(distance)})
    return hits
