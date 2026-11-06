from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import settings

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load one configurable multilingual model per process."""
    return SentenceTransformer(settings.embedding_model)

def add_task_prefix(texts: list[str], *, is_query: bool) -> list[str]:
    """E5 models need "query: " on questions and "passage: " on documents."""
    if "-e5-" not in settings.embedding_model:
        return texts
    prefix = "query: " if is_query else "passage: "
    return [prefix + text for text in texts]

def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedder().encode(
        add_task_prefix(texts, is_query=is_query),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
