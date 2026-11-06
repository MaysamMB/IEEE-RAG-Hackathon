"""Evaluation scoring tests; retrieval and the LLM are faked."""
import json

import pytest

from rag_hackathon import evaluation
from rag_hackathon.evaluation import (
    evaluate,
    is_refusal,
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


def test_no_sources_counts_as_refusal():
    assert is_refusal("أي نص", [])


def test_refusal_wording_is_detected_with_letter_variants():
    answer = "لا تحتوي الوثائق المتاحة على معلومات كافية للإجابة."

    assert is_refusal(answer, [{"source": "guide.pdf", "page": 1}])


def test_caveat_after_a_real_answer_is_not_a_refusal():
    answer = "الرسوم 480 ديناراً حسب المادة الخامسة من الدليل، وتدفع قبل نهاية الأسبوع الثاني من الفصل الدراسي كما ورد في المصدر [S1]. " + "لا تحتوي الوثائق على رسوم أخرى."

    assert not is_refusal(answer, [{"source": "guide.pdf", "page": 1}])


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("الرسوم 250 دينار [S1]", [{"source": "guide.pdf", "page": 1}])


def test_correct_answer_scores():
    scores = score_answer(question(), "تبلغ الرسوم 250 دينار [S1]", [{"source": "docs/guide.pdf", "page": 3}])

    assert scores["facts_correct"] is True
    assert scores["citation_correct"] is True
    assert scores["refusal_correct"] is True


def test_fact_accepts_alternative_arabic_endings():
    item = question(expected_answer_facts=["فصلين|فصلان"])

    scores = score_answer(item, "الحد هو فصلان دراسيان", [{"source": "guide.pdf", "page": 3}])

    assert scores["facts_correct"] is True


def test_missing_fact_and_wrong_page_are_reported():
    scores = score_answer(question(), "الرسوم مرتفعة [S1]", [{"source": "guide.pdf", "page": 9}])

    assert scores["missing_facts"] == ["250 دينار"]
    assert scores["citation_correct"] is False


def test_similar_file_name_is_not_a_correct_citation():
    scores = score_answer(question(), "250 دينار", [{"source": "my_guide.pdf", "page": 3}])

    assert scores["citation_correct"] is False


def test_unwanted_refusal_is_wrong():
    scores = score_answer(question(), "لا تحتوي الوثائق على معلومات كافية", [{"source": "guide.pdf", "page": 3}])

    assert scores["refusal_correct"] is False


def test_answering_an_out_of_scope_question_is_wrong():
    scores = score_answer(question(should_refuse=True, expected_sources=[]), "فاز الفريق", [{"source": "guide.pdf", "page": 1}])

    assert scores["refusal_correct"] is False
    assert "facts_correct" not in scores


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


def test_failed_question_is_recorded_and_the_run_continues(monkeypatch):
    def fail(_):
        raise RuntimeError("network down")

    monkeypatch.setattr(evaluation, "retrieve", lambda q: [hit(3)])
    monkeypatch.setattr(evaluation, "answer_question", fail)

    report = evaluate([question(id="Q1"), question(id="Q2")], with_answers=True)

    assert report["summary"]["errors"] == 2
    assert "network down" in report["rows"][1]["error"]
