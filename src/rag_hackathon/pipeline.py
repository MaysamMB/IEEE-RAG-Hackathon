from .retrieval import retrieve
from .generation import cited_numbers, generate_answer, rewrite_question, source_label

def answer_question(question: str, history: list[dict] | tuple = ()) -> dict:
    """history holds earlier {"question", "answer"} turns of this conversation."""
    # Documents are formal, so even a first question is searched in formal wording.
    rewritten = rewrite_question(question, list(history))
    search_question = rewritten.text
    hits = retrieve(search_question)
    generated = generate_answer(search_question, hits)
    sources = []
    # A refusal rests on no evidence, so it must not display any source.
    for index, hit in enumerate([] if generated.refused else hits, start=1):
        metadata = hit["metadata"]
        sources.append({
            "id": f"S{index}",
            "source": source_label(metadata),
            "page": metadata.get("page", "Unknown"),
            "article": metadata.get("article"),
            "file_type": metadata.get("file_type"),
            "distance": hit["distance"],
        })
    # IDs stay as the model saw them, so [S2] in the answer still points to S2.
    cited = cited_numbers(generated.text)
    used_sources = [source for source in sources if int(source["id"][1:]) in cited]
    return {
        "answer": generated.user_text,
        "refused": generated.refused,
        "sources": used_sources or sources,
        "retrieved_count": len(hits),
        "search_question": search_question,
        "input_tokens": generated.input_tokens + rewritten.input_tokens,
        "output_tokens": generated.output_tokens + rewritten.output_tokens,
    }
