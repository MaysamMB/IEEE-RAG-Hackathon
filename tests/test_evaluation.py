"""Evaluation scoring tests; retrieval and the LLM are faked."""
import json
from types import SimpleNamespace

import pytest

from rag_hackathon import evaluation
from rag_hackathon.evaluation import (
    evaluate,
    load_questions,
    normalize_text,
    score_answer,
    score_retrieval,
    source_matches,
    summarize,
)


def question(**overrides):
    item = {
        "id": "Q1",
        "category": "direct_fact",
        "question": "كم الرسوم؟",
        "expected_answer_facts": ["250 دينار"],
        "expected_sources": [{"source": "guide.pdf", "page": 3}],
        "should_refuse": False,
    }
    return {**item, **overrides}


def hit(page, distance=0.2):
    return {
        "text": "evidence",
        "metadata": {"source": "guide.pdf", "source_path": "docs/guide.pdf", "page": page},
        "distance": distance,
    }


@pytest.fixture(autouse=True)
def fake_index(monkeypatch):
    collection = SimpleNamespace(
        metadata={"embedding_model": "mock", "chunk_size": 1000, "chunk_overlap": 150, "documents_root": "/docs"},
        count=lambda: 22,
    )
    monkeypatch.setattr(evaluation, "get_collection", lambda: collection)


def write_questions(tmp_path, items):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def test_normalization_ignores_diacritics_and_letter_variants():
    assert normalize_text("الإجابةُ  مُؤكَّدة") == normalize_text("الاجابه مؤكده")


def test_template_placeholders_are_rejected(tmp_path):
    path = write_questions(tmp_path, [question(question="REPLACE_WITH_A_REAL_QUESTION")])

    with pytest.raises(ValueError, match="placeholders"):
        load_questions(path)


def test_shipped_template_cannot_be_scored_by_mistake():
    with pytest.raises(ValueError, match="placeholders"):
        load_questions("evaluation/questions.template.json")


def test_missing_field_is_rejected(tmp_path):
    item = question()
    del item["should_refuse"]

    with pytest.raises(ValueError, match="should_refuse"):
        load_questions(write_questions(tmp_path, [item]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        load_questions(write_questions(tmp_path, [question(), question()]))


def test_source_matches_by_file_name_or_relative_path():
    metadata = hit(3)["metadata"]

    assert source_matches(metadata, {"source": "guide.pdf", "page": 3})
    assert source_matches(metadata, {"source": "docs/guide.pdf", "page": 3})
    assert source_matches(metadata, {"source": "guide.pdf"})
    assert not source_matches(metadata, {"source": "guide.pdf", "page": 4})
    assert not source_matches(metadata, {"source": "other.pdf", "page": 3})


def test_retrieval_scores_rank_precision_and_recall(monkeypatch):
    monkeypatch.setattr(evaluation, "settings", type("S", (), {"top_k": 4}))
    scores = score_retrieval(question(), [hit(9), hit(3), hit(8), hit(7)])

    assert scores["hit"] is True
    assert scores["first_rank"] == 2
    assert scores["precision"] == 0.25
    assert scores["recall"] == 1


def test_precision_uses_k_even_with_fewer_hits(monkeypatch):
    monkeypatch.setattr(evaluation, "settings", type("S", (), {"top_k": 4}))

    assert score_retrieval(question(), [hit(3)])["precision"] == 0.25


def test_retrieval_miss_scores_zero(monkeypatch):
    monkeypatch.setattr(evaluation, "settings", type("S", (), {"top_k": 4}))
    scores = score_retrieval(question(), [hit(9)])

    assert scores["hit"] is False
    assert scores["first_rank"] is None
    assert scores["recall"] == 0


def test_recall_counts_each_expected_source():
    scores = score_retrieval(
        question(expected_sources=[{"source": "guide.pdf", "page": 3}, {"source": "guide.pdf", "page": 4}]),
        [hit(3)],
    )

    assert scores["recall"] == 0.5


def test_out_of_scope_question_has_no_hit_score():
    assert score_retrieval(question(expected_sources=[], should_refuse=True), [hit(1, 0.9)]) == {"top_distance": 0.9}


def cited(*items):
    return [{"id": f"S{number}", "source": name, "page": page} for number, name, page in items]


def test_correct_answer_scores():
    scores = score_answer(question(), "تبلغ الرسوم 250 دينار [S1]", cited((1, "docs/guide.pdf", 3)), False)

    assert scores["facts_correct"] is True
    assert scores["citation_correct"] is True
    assert scores["refusal_correct"] is True


def test_fact_accepts_alternative_arabic_endings():
    item = question(expected_answer_facts=["فصلين|فصلان"])

    scores = score_answer(item, "الحد هو فصلان دراسيان [S1]", cited((1, "guide.pdf", 3)), False)

    assert scores["facts_correct"] is True


def test_missing_fact_is_reported():
    scores = score_answer(question(), "الرسوم مرتفعة [S1]", cited((1, "guide.pdf", 3)), False)

    assert scores["missing_facts"] == ["250 دينار"]


def test_right_source_retrieved_but_not_cited_is_wrong():
    sources = cited((1, "other.pdf", 1), (2, "guide.pdf", 3))

    scores = score_answer(question(), "250 دينار [S1]", sources, False)

    assert scores["citation_correct"] is False
    assert scores["cited_sources"] == ["other.pdf p1"]


def test_cited_right_source_among_retrieved_ones_is_correct():
    sources = cited((1, "other.pdf", 1), (2, "guide.pdf", 3))

    assert score_answer(question(), "250 دينار [S2]", sources, False)["citation_correct"] is True


def test_answer_without_citation_is_wrong():
    scores = score_answer(question(), "250 دينار", cited((1, "guide.pdf", 3)), False)

    assert scores["citation_correct"] is False


def test_answer_needs_a_citation_even_without_expected_sources():
    scores = score_answer(question(expected_sources=[]), "250 دينار", cited((1, "guide.pdf", 3)), False)

    assert scores["citation_correct"] is False


def test_citing_a_source_that_does_not_exist_is_wrong():
    scores = score_answer(question(), "250 دينار [S1] [S7]", cited((1, "guide.pdf", 3)), False)

    assert scores["invalid_citations"] == ["S7"]
    assert scores["citation_correct"] is False


def test_every_expected_source_must_be_cited():
    item = question(expected_sources=[{"source": "a.txt", "page": 1}, {"source": "b.txt", "page": 1}])
    sources = cited((1, "a.txt", 1), (2, "b.txt", 1))

    assert score_answer(item, "250 دينار [S1]", sources, False)["citation_correct"] is False
    assert score_answer(item, "250 دينار [S1][S2]", sources, False)["citation_correct"] is True


def test_wrong_page_is_a_wrong_citation():
    scores = score_answer(question(), "250 دينار [S1]", cited((1, "guide.pdf", 9)), False)

    assert scores["citation_correct"] is False


def test_similar_file_name_is_not_a_correct_citation():
    scores = score_answer(question(), "250 دينار [S1]", cited((1, "my_guide.pdf", 3)), False)

    assert scores["citation_correct"] is False


def test_unwanted_refusal_is_wrong():
    scores = score_answer(question(), "لا تحتوي", [], True)

    assert scores["refusal_correct"] is False
    assert scores["citation_correct"] is False


def test_answering_an_out_of_scope_question_is_wrong():
    scores = score_answer(question(should_refuse=True, expected_sources=[]), "فاز الفريق", cited((1, "guide.pdf", 1)), False)

    assert scores["refusal_correct"] is False
    assert "facts_correct" not in scores


def test_correct_refusal_scores():
    scores = score_answer(question(should_refuse=True, expected_sources=[]), "لا تحتوي", [], True)

    assert scores["refusal_correct"] is True


def test_summary_separates_in_scope_and_out_of_scope_distances():
    rows = [
        {"category": "direct", "hit": True, "first_rank": 2, "precision": 0.25, "recall": 1, "top_distance": 0.4},
        {"category": "direct", "hit": False, "first_rank": None, "precision": 0, "recall": 0, "top_distance": 0.6},
        {"category": "out", "top_distance": 0.9},
    ]

    summary = summarize(rows)

    assert summary["hit_at_k"] == {"passed": 1, "total": 2}
    assert summary["mrr"] == 0.25
    assert summary["largest_in_scope_distance"] == 0.6
    assert summary["smallest_out_of_scope_distance"] == 0.9
    assert summary["by_category"]["direct"]["hit_at_k"] == {"passed": 1, "total": 2}


def test_usage_is_averaged_over_real_model_calls_only():
    rows = [
        {"input_tokens": 100, "output_tokens": 10},
        {"input_tokens": 300, "output_tokens": 30},
        {"input_tokens": 0, "output_tokens": 0},
    ]

    usage = evaluation.usage_summary(rows)

    assert usage["model_calls"] == 2
    assert usage["average_input_tokens"] == 200
    assert usage["average_output_tokens"] == 20


def test_cost_stays_empty_without_prices(monkeypatch):
    monkeypatch.setattr(evaluation, "settings", type("S", (), {"input_price_per_mtok": None, "output_price_per_mtok": None}))

    assert evaluation.usage_summary([{"input_tokens": 1000, "output_tokens": 100}])["average_cost_usd"] is None


def test_cost_uses_configured_prices(monkeypatch):
    monkeypatch.setattr(evaluation, "settings", type("S", (), {"input_price_per_mtok": 2.0, "output_price_per_mtok": 10.0}))

    assert evaluation.usage_summary([{"input_tokens": 1000, "output_tokens": 100}])["average_cost_usd"] == 0.003


def test_usage_is_absent_without_model_calls():
    assert evaluation.usage_summary([{"id": "Q1"}]) is None


def test_evaluate_without_answers_never_calls_the_llm(monkeypatch):
    def forbidden(_):
        raise AssertionError("the paid LLM must not be called")

    monkeypatch.setattr(evaluation, "retrieve", lambda q: [hit(3)])
    monkeypatch.setattr(evaluation, "answer_question", forbidden)

    report = evaluate([question()])

    assert report["summary"]["hit_at_k"] == {"passed": 1, "total": 1}
    assert "answer" not in report["rows"][0]
    assert "error" not in report["rows"][0]


def test_role_based_history_becomes_turns():
    item = question(conversation_history=[
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])

    assert evaluation.conversation_turns(item) == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": ""},
    ]


def test_question_without_history_has_no_turns():
    assert evaluation.conversation_turns(question()) == []


def test_follow_up_is_skipped_without_answers_and_never_calls_the_llm(monkeypatch):
    def forbidden(*_):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(evaluation, "retrieve", forbidden)
    monkeypatch.setattr(evaluation, "answer_question", forbidden)

    report = evaluate([question(history=[{"question": "q", "answer": "a"}])])

    assert "skipped" in report["rows"][0]
    assert report["summary"]["hit_at_k"] is None


def test_follow_up_retrieval_is_scored_on_the_rewritten_question(monkeypatch):
    searched = []

    def fake_retrieve(text):
        searched.append(text)
        return [hit(3)]

    history = [{"question": "q", "answer": "a"}]
    monkeypatch.setattr(evaluation, "retrieve", fake_retrieve)
    monkeypatch.setattr(evaluation, "answer_question", lambda text, turns: {
        "answer": "250 دينار [S1]", "refused": False, "search_question": "full question",
        "input_tokens": 1, "output_tokens": 1,
        "sources": [{"id": "S1", "source": "guide.pdf", "page": 3}],
    })

    report = evaluate([question(history=history)], with_answers=True)

    assert searched == ["full question"]
    assert report["rows"][0]["search_question"] == "full question"
    assert report["rows"][0]["hit"] is True


def test_failed_question_is_recorded_and_the_run_continues(monkeypatch):
    def fail(*_):
        raise RuntimeError("network down")

    monkeypatch.setattr(evaluation, "retrieve", lambda q: [hit(3)])
    monkeypatch.setattr(evaluation, "answer_question", fail)

    report = evaluate([question(id="Q1"), question(id="Q2")], with_answers=True)

    assert report["summary"]["errors"] == 2
    assert "network down" in report["rows"][1]["error"]


def test_report_records_what_produced_the_result(monkeypatch):
    monkeypatch.setattr(evaluation, "retrieve", lambda q: [hit(3)])

    used = evaluate([question()])["settings"]

    assert used["embedding_model"] == "mock"
    assert (used["chunk_size"], used["chunk_overlap"]) == (1000, 150)
    assert used["documents_root"] == "/docs"
    assert used["indexed_chunks"] == 22
    assert used["top_k"] == evaluation.settings.top_k


def test_report_works_without_an_index(monkeypatch):
    monkeypatch.setattr(evaluation, "get_collection", lambda: None)

    assert evaluation.run_settings()["indexed_chunks"] == 0
