from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "intfloat/multilingual-e5-small",
    )
    ocr_cache_dir: str = os.getenv("OCR_CACHE_DIR", "data/processed/ocr_cache")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "storage/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "hackathon_documents")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    input_price_per_mtok: float | None = (
        float(os.environ["INPUT_PRICE_PER_MTOK"])
        if os.getenv("INPUT_PRICE_PER_MTOK", "").strip()
        else None
    )
    output_price_per_mtok: float | None = (
        float(os.environ["OUTPUT_PRICE_PER_MTOK"])
        if os.getenv("OUTPUT_PRICE_PER_MTOK", "").strip()
        else None
    )
    max_cosine_distance: float | None = (
        float(os.environ["MAX_COSINE_DISTANCE"])
        if os.getenv("MAX_COSINE_DISTANCE", "").strip()
        else None
    )

settings = Settings()
