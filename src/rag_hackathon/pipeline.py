from .retrieval import retrieve
from .generation import generate_answer, source_label

def answer_question(question: str) -> dict:
    hits = retrieve(question)
    generated = generate_answer(question, hits)
    sources = []
    # A refusal rests on no evidence, so it must not display any source.
    for index, hit in enumerate([] if generated.refused else hits, start=1):
        metadata = hit["metadata"]
        sources.append({
            "id": f"S{index}",
            "source": source_label(metadata),
            "page": metadata.get("page", "Unknown"),
            "distance": hit["distance"],
        })
    return {
        "answer": generated.text,
        "refused": generated.refused,
        "sources": sources,
        "retrieved_count": len(hits),
        "input_tokens": generated.input_tokens,
        "output_tokens": generated.output_tokens,
    }
