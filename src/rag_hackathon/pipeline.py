from .retrieval import retrieve
from .generation import generate_answer

def answer_question(question: str) -> dict:
    hits = retrieve(question)
    answer = generate_answer(question, hits)
    sources = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        sources.append({
            "id": f"S{index}",
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": hit["distance"],
        })
    return {"answer": answer, "sources": sources, "retrieved_count": len(hits)}
