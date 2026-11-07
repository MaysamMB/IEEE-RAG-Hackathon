"""Measure retrieval, answers, citations and refusal on a set of labelled questions."""
import json
from pathlib import Path
import re
from statistics import mean
import unicodedata

from .config import settings
from .pipeline import answer_question
from .retrieval import retrieve
from .vector_store import get_collection

PLACEHOLDER = re.compile(r"REPLACE_|REWORD_|ASK_FOR_")
MINIMUM_QUESTIONS = 10
LETTER_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})
CITED_ID = re.compile(r"\bS(\d+)\b")
ARABIC_MARKS = re.compile("[ً-ْـ]")

def normalize_text(text: str) -> str:
    """Make text comparison tolerant of diacritics and common letter variants."""
    text = ARABIC_MARKS.sub("", unicodedata.normalize("NFKC", text))
    return " ".join(text.translate(LETTER_VARIANTS).lower().split())

def load_questions(path: str | Path) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    if PLACEHOLDER.search(raw):
        raise ValueError(f"{path} still contains template placeholders.")
    questions = json.loads(raw)
    required = {"id", "category", "question", "expected_answer_facts", "expected_sources", "should_refuse"}
    for item in questions:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Question {item.get('id', '?')} is missing {sorted(missing)}.")
    ids = [item["id"] for item in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique.")
    return questions

def conversation_turns(question: dict) -> list[dict]:
    """Read earlier turns from either "history" or role-based "conversation_history"."""
    turns = list(question.get("history", []))
    for message in question.get("conversation_history", []):
        if message["role"] == "user":
            turns.append({"question": message["content"], "answer": ""})
        elif turns:
            turns[-1]["answer"] = message["content"]
    return turns

def source_matches(metadata: dict, expected: dict) -> bool:
    name = expected["source"]
    same_file = name in (metadata.get("source_path"), metadata.get("source"))
    return same_file and metadata.get("page") == expected.get("page", metadata.get("page"))

def score_retrieval(question: dict, hits: list[dict]) -> dict:
    expected = question["expected_sources"]
    top_distance = hits[0]["distance"] if hits else None
    if not expected:
        return {"top_distance": top_distance}
    relevant = [any(source_matches(h["metadata"], e) for e in expected) for h in hits]
    found = [e for e in expected if any(source_matches(h["metadata"], e) for h in hits)]
    first = next((rank for rank, is_relevant in enumerate(relevant, start=1) if is_relevant), None)
    return {
        "top_distance": top_distance,
        "hit": first is not None,
        "first_rank": first,
        "precision": sum(relevant) / settings.top_k,
        "recall": len(found) / len(expected),
    }

def fact_found(fact: str, normalized_answer: str) -> bool:
    """A fact may list spellings with "|", because Arabic endings change with grammar."""
    return any(normalize_text(option) in normalized_answer for option in fact.split("|"))

def _is_same_citation(source: dict, expected: dict) -> bool:
    same_file = source["source"] == expected["source"] or source["source"].endswith("/" + expected["source"])
    return same_file and source["page"] == expected.get("page", source["page"])

def citation_scores(question: dict, answer: str, sources: list[dict]) -> dict:
    """Judge the sources the answer actually cites, not everything retrieved."""
    cited_ids = {f"S{number}" for number in CITED_ID.findall(answer)}
    cited = [source for source in sources if source["id"] in cited_ids]
    invalid = sorted(cited_ids - {source["id"] for source in sources})
    covered = all(
        any(_is_same_citation(source, expected) for source in cited)
        for expected in question["expected_sources"]
    )
    return {
        "cited_sources": [f"{source['source']} p{source['page']}" for source in cited],
        "invalid_citations": invalid,
        "citation_correct": bool(cited) and covered and not invalid,
    }

def score_answer(question: dict, answer: str, sources: list[dict], refused: bool) -> dict:
    scores = {"refused": refused, "refusal_correct": refused == question["should_refuse"]}
    if question["should_refuse"]:
        return scores
    text = normalize_text(answer)
    facts = question["expected_answer_facts"]
    scores["missing_facts"] = [fact for fact in facts if not fact_found(fact, text)]
    scores["facts_correct"] = not scores["missing_facts"]
    scores.update(citation_scores(question, answer, sources))
    return scores

def usage_summary(rows: list[dict]) -> dict | None:
    """Average tokens per model call; cost appears only when prices are configured."""
    calls = [row for row in rows if row.get("input_tokens")]
    if not calls:
        return None
    input_tokens = mean(row["input_tokens"] for row in calls)
    output_tokens = mean(row["output_tokens"] for row in calls)
    usage = {
        "model_calls": len(calls),
        "average_input_tokens": round(input_tokens),
        "average_output_tokens": round(output_tokens),
        "average_cost_usd": None,
    }
    if settings.input_price_per_mtok is not None and settings.output_price_per_mtok is not None:
        cost = input_tokens * settings.input_price_per_mtok + output_tokens * settings.output_price_per_mtok
        usage["average_cost_usd"] = round(cost / 1_000_000, 6)
    return usage

def _share(rows: list[dict], key: str) -> dict | None:
    values = [row[key] for row in rows if key in row]
    return {"passed": sum(values), "total": len(values)} if values else None

def summarize(rows: list[dict]) -> dict:
    ranked = [row for row in rows if "hit" in row]
    in_scope = [r["top_distance"] for r in ranked if r["top_distance"] is not None]
    out_of_scope = [
        r["top_distance"] for r in rows
        if "hit" not in r and r["top_distance"] is not None
    ]
    summary = {
        "questions": len(rows),
        "top_k": settings.top_k,
        "hit_at_k": _share(rows, "hit"),
        "precision_at_k": round(mean(r["precision"] for r in ranked), 3) if ranked else None,
        "recall_at_k": round(mean(r["recall"] for r in ranked), 3) if ranked else None,
        "mrr": round(mean(1 / r["first_rank"] if r["first_rank"] else 0 for r in ranked), 3) if ranked else None,
        "largest_in_scope_distance": round(max(in_scope), 3) if in_scope else None,
        "smallest_out_of_scope_distance": round(min(out_of_scope), 3) if out_of_scope else None,
        "answers_with_all_facts": _share(rows, "facts_correct"),
        "correct_citations": _share(rows, "citation_correct"),
        "correct_refusal_decisions": _share(rows, "refusal_correct"),
        "usage": usage_summary(rows),
        "errors": sum("error" in row for row in rows),
        "by_category": {},
    }
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        summary["by_category"][category] = {
            "hit_at_k": _share(group, "hit"),
            "correct_refusal_decisions": _share(group, "refusal_correct"),
        }
    return summary

def run_settings() -> dict:
    """Record what produced a result, read from the index itself where possible."""
    collection = get_collection()
    metadata = collection.metadata if collection is not None else {}
    return {
        "embedding_model": metadata.get("embedding_model"),
        "chunk_size": metadata.get("chunk_size"),
        "chunk_overlap": metadata.get("chunk_overlap"),
        "documents_root": metadata.get("documents_root"),
        "indexed_chunks": collection.count() if collection is not None else 0,
        "top_k": settings.top_k,
        "max_cosine_distance": settings.max_cosine_distance,
        "generation_model": settings.llm_model,
    }

def evaluate(questions: list[dict], *, with_answers: bool = False) -> dict:
    """Score every question. with_answers calls the LLM twice per question."""
    rows = []
    for question in questions:
        row = {"id": question["id"], "category": question["category"], "question": question["question"]}
        history = conversation_turns(question)
        # A follow-up only makes sense after rewriting, which needs the model.
        if history and not with_answers:
            row.update({"top_distance": None, "skipped": "follow-up: run with --answers"})
            rows.append(row)
            continue
        result, failure = None, None
        if with_answers:
            try:
                result = answer_question(question["question"], history)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
        search_question = result.get("search_question", question["question"]) if result else question["question"]
        if result or history:
            row["search_question"] = search_question
        hits = retrieve(search_question)
        row.update(score_retrieval(question, hits))
        row["retrieved"] = [
            {"source": h["metadata"].get("source_path"), "page": h["metadata"].get("page"), "distance": round(h["distance"], 3)}
            for h in hits
        ]
        if failure:
            row["error"] = failure
        elif result:
            row["answer"] = result["answer"]
            row["input_tokens"] = result["input_tokens"]
            row["output_tokens"] = result["output_tokens"]
            row.update(score_answer(question, result["answer"], result["sources"], result["refused"]))
        rows.append(row)
    return {"settings": run_settings(), "summary": summarize(rows), "rows": rows}
