from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .config import settings

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load one configurable multilingual model per process."""
    return SentenceTransformer(settings.embedding_model)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
