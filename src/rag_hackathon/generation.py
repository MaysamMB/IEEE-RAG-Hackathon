from dataclasses import dataclass
from anthropic import Anthropic
from .config import settings

SYSTEM_PROMPT = """You are an evidence-grounded assistant.
Answer the user's question using only the supplied retrieved context.
Do not use outside knowledge to fill gaps. If the context is insufficient,
state clearly that the available documents do not contain enough information.
Cite supporting source identifiers exactly as [S1], [S2], etc.
Do not invent page numbers, sources, facts, or quotations. If sources conflict,
explain the conflict and cite each relevant source. Answer in the user's language.
"""

REFUSAL_MESSAGE = "لا تحتوي المستندات المتاحة على معلومات كافية للإجابة عن هذا السؤال."

@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

def source_label(metadata: dict) -> str:
    """Prefer the relative path: file names alone are ambiguous across folders."""
    return metadata.get("source_path") or metadata.get("source") or "Unknown"

def generate_answer(question: str, hits: list[dict]) -> GeneratedAnswer:
    # Refusing needs no model call, so it costs nothing and works without a key.
    if not hits:
        return GeneratedAnswer(REFUSAL_MESSAGE)
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Set it in your local .env file.")
    if not settings.anthropic_model:
        raise RuntimeError("ANTHROPIC_MODEL is missing. Set a model available to your API account in .env.")

    context_parts = []
    for index, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        context_parts.append(
            f"[S{index}] Source: {source_label(meta)}; "
            f"Page: {meta.get('page', 'Unknown')}\n{hit['text']}"
        )
    prompt = (
        "Retrieved context:\n\n" + "\n\n---\n\n".join(context_parts)
        + f"\n\nUser question: {question}\n\nAnswer with source IDs."
    )
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return GeneratedAnswer(text, response.usage.input_tokens, response.usage.output_tokens)
