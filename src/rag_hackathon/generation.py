from dataclasses import dataclass, replace
import logging
import re
from openai import OpenAI, RateLimitError
from .config import settings

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "لا تحتوي المستندات المتاحة على معلومات كافية للإجابة عن هذا السؤال."
REFERRAL_MESSAGE = "يرجى مراجعة عمادة القبول والتسجيل في الجامعة."
OUT_OF_SCOPE_MESSAGE = "عذراً، أنا مرشد أكاديمي وأجيب فقط عن الأسئلة الأكاديمية الواردة في مستندات الجامعة."
MAX_HISTORY_TURNS = 3

# Spaces and invisible direction marks models sometimes place inside a citation.
_CITATION_NOISE = r"[\s​-‏‪-‮⁠-⁩﻿]*"
CITATION_VARIANT = re.compile(
    rf"[\[［【]{_CITATION_NOISE}[Ssس]{_CITATION_NOISE}(\d+){_CITATION_NOISE}[\]］】]"
)
CITED_ID = re.compile(r"\[S(\d+)\]")

SYSTEM_PROMPT = f"""You are an evidence-grounded academic advisor for university students.
If the question is not about university study, reply with exactly this sentence
and nothing else: {OUT_OF_SCOPE_MESSAGE}
Otherwise answer using only the supplied retrieved context.
Do not use outside knowledge to fill gaps. If the context is insufficient,
reply with exactly this sentence and nothing else: {REFUSAL_MESSAGE}
Cite supporting source identifiers exactly as [S1], [S2], etc., always with the
Latin letter S and digits, even inside Arabic text, and with no spaces or other
characters inside the brackets. Never write the identifier with an Arabic letter.
Do not invent page numbers, sources, facts, or quotations.
Values that apply to different programs or student groups are not a conflict.
If the question names no program, answer for the bachelor's degree and do not
use diploma rules.
If sources give different values for the same thing, adopt the value from the
source with the latest Version as the answer, then mention the older value and
its Version in one short sentence, citing both sources. If the Versions are
equal, adopt the file named regulations or instructions, because it is the
binding reference, and say so. If a conflicting source has no Version, do not
choose: explain the conflict and cite each source.
Answer in simple Modern Standard Arabic, even when the question is in dialect.
Write plain text only: never use LaTeX or math markup. Write calculations
like this: 45 × 18 = 810.
"""

REWRITE_PROMPT = """Rewrite the student's latest question as one complete, standalone
question in simple Modern Standard Arabic, using the formal terms found in university
regulations in place of dialect words. Use the conversation, if any, only to resolve
what the question refers to. Do not answer it. Keep every number as it is.
If a question about study rules names no program, ask it about the bachelor's degree.
Apart from that, do not add facts or conditions the student did not state.
If it is not about university study, return it unchanged.
Return only the question.
"""

@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def refused(self) -> bool:
        return self.text.strip(" *\n").startswith((REFUSAL_MESSAGE, OUT_OF_SCOPE_MESSAGE))

    @property
    def user_text(self) -> str:
        """Refusals come from fixed text, so the referral is always present and exact."""
        stripped = self.text.strip(" *\n")
        if stripped.startswith(REFUSAL_MESSAGE):
            return f"{REFUSAL_MESSAGE} {REFERRAL_MESSAGE}"
        if stripped.startswith(OUT_OF_SCOPE_MESSAGE):
            return OUT_OF_SCOPE_MESSAGE
        return self.text

def normalize_citations(text: str) -> str:
    """Rewrite model spellings such as [ S1 ], 【S1】 or [س١] as [S1]."""
    return CITATION_VARIANT.sub(lambda match: f"[S{int(match.group(1))}]", text)

def cited_numbers(text: str) -> set[int]:
    return {int(number) for number in CITED_ID.findall(text)}

def source_label(metadata: dict) -> str:
    """Prefer the relative path: file names alone are ambiguous across folders."""
    return metadata.get("source_path") or metadata.get("source") or "Unknown"

def _call_model(client: OpenAI, model: str, system: str, user: str, max_tokens: int):
    # Reasoning settings suit the main model; other models may reject them.
    effort = settings.llm_reasoning_effort if model == settings.llm_model else ""
    if model == "qwen/qwen3.8-27b":
        max_tokens = min(max_tokens, 700)
    return client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **({"extra_body": {"reasoning_effort": effort}} if effort else {}),
    )

def _chat(system: str, user: str, max_tokens: int, model: str = "") -> GeneratedAnswer:
    # The base URL is required: left blank, the SDK would default to OpenAI's paid API.
    missing = [
        name for name, value in (
            ("LLM_API_KEY", settings.llm_api_key),
            ("LLM_BASE_URL", settings.llm_base_url),
            ("LLM_MODEL", settings.llm_model),
        ) if not value
    ]
    if missing:
        raise RuntimeError(f"Missing in your local .env file: {', '.join(missing)}.")
    # Free tiers answer 429 when busy; the SDK waits and retries those calls.
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_retries=3,
        timeout=60,
    )
    primary = model or settings.llm_model
    try:
        response = _call_model(client, primary, system, user, max_tokens)
    except RateLimitError:
        fallback = settings.llm_fallback_model
        if not fallback or fallback == primary:
            raise
        # Quotas are per model, so a demo can continue when the main one runs out.
        logger.warning("Quota reached for %s; using %s", primary, fallback)
        response = _call_model(client, fallback, system, user, max_tokens)
    text = (response.choices[0].message.content or "").strip() if response.choices else ""
    # An empty reply must fail loudly: showing it would look like a blank answer.
    if not text:
        raise RuntimeError("The model returned an empty answer.")
    usage = response.usage
    return GeneratedAnswer(text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)

def rewrite_question(question: str, history: list[dict]) -> GeneratedAnswer:
    """Turn dialect or a follow-up into a formal standalone question for retrieval."""
    turns = "\n".join(
        f"Student: {turn['question']}\nAdvisor: {turn['answer']}"
        for turn in history[-MAX_HISTORY_TURNS:]
    )
    prompt = f"Conversation:\n{turns or '(none)'}\n\nLatest question: {question}"
    try:
        return _chat(REWRITE_PROMPT, prompt, max_tokens=300, model=settings.llm_rewrite_model)
    except Exception:
        # A failed rewrite should not lose the answer: fall back to the question as asked.
        logger.exception("Question rewrite failed")
        return GeneratedAnswer(question)

def generate_answer(question: str, hits: list[dict]) -> GeneratedAnswer:
    # Refusing needs no model call, so it costs nothing and works without a key.
    if not hits:
        return GeneratedAnswer(REFUSAL_MESSAGE)
    context_parts = []
    for index, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        article = f"; Article: {meta['article']}" if meta.get("article") else ""
        version = f"; Version: {meta['version']}" if meta.get("version") else ""
        context_parts.append(
            f"[S{index}] Source: {source_label(meta)}; "
            f"Page: {meta.get('page', 'Unknown')}{article}{version}\n{hit['text']}"
        )
    prompt = (
        "Retrieved context:\n\n" + "\n\n---\n\n".join(context_parts)
        + f"\n\nUser question: {question}\n\nAnswer with source IDs."
    )
    # Reasoning models spend part of this limit before writing, so 700 could leave nothing.
    answer = _chat(SYSTEM_PROMPT, prompt, max_tokens=1500)
    return replace(answer, text=normalize_citations(answer.text))
