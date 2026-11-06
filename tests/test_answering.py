"""Citation and user-facing error tests; no model, API key or network is needed."""
from dataclasses import replace
import importlib.util
import logging
from pathlib import Path
import runpy
from types import SimpleNamespace

import gradio as gr
import pytest

from rag_hackathon import config, generation, pipeline
from rag_hackathon.vector_store import EmbeddingModelMismatchError, InterruptedRebuildError

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_hit(path, name="policy.pdf", page=3):
    return {
        "text": "evidence",
        "metadata": {"source": name, "source_path": path, "page": page},
        "distance": 0.1,
    }


@pytest.fixture
def app():
    spec = importlib.util.spec_from_file_location("demo_app", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_answer_sources_distinguish_same_named_files(monkeypatch):
    hits = [make_hit("2023/policy.pdf"), make_hit("2024/policy.pdf")]
    monkeypatch.setattr(pipeline, "retrieve", lambda question: hits)
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, found: generation.GeneratedAnswer("answer", 11, 7))

    result = pipeline.answer_question("question")

    assert [s["source"] for s in result["sources"]] == ["2023/policy.pdf", "2024/policy.pdf"]
    assert (result["input_tokens"], result["output_tokens"]) == (11, 7)


def test_prompt_cites_relative_paths(monkeypatch):
    sent = {}

    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="answer")],
                usage=SimpleNamespace(input_tokens=120, output_tokens=30),
            )

    monkeypatch.setattr(generation, "Anthropic", FakeClient)
    monkeypatch.setattr(
        generation, "settings", replace(config.settings, anthropic_api_key="test", anthropic_model="test")
    )

    answer = generation.generate_answer("question", [make_hit("2024/policy.pdf")])

    assert "Source: 2024/policy.pdf; Page: 3" in sent["messages"][0]["content"]
    assert answer == generation.GeneratedAnswer("answer", 120, 30)


def test_refusal_without_evidence_needs_no_key_or_model_call(monkeypatch):
    def forbidden(api_key):
        raise AssertionError("no model call is needed")

    monkeypatch.setattr(generation, "Anthropic", forbidden)
    monkeypatch.setattr(generation, "settings", replace(config.settings, anthropic_api_key="", anthropic_model=""))

    assert generation.generate_answer("question", []) == generation.GeneratedAnswer(generation.REFUSAL_MESSAGE)


def test_source_label_falls_back_for_older_metadata():
    assert generation.source_label({"source": "policy.pdf"}) == "policy.pdf"
    assert generation.source_label({}) == "Unknown"


def test_page_shows_relative_path_as_citation(monkeypatch, app):
    answer = {
        "answer": "answer",
        "sources": [{"id": "S1", "source": "2024/policy.pdf", "page": 3, "distance": 0.1}],
    }
    monkeypatch.setattr(app, "answer_question", lambda question: answer)

    assert "2024/policy.pdf" in app.ask("question")


@pytest.mark.parametrize("error", [
    InterruptedRebuildError("internal rebuild detail"),
    EmbeddingModelMismatchError("internal model detail"),
])
def test_index_errors_show_arabic_message_and_log_details(monkeypatch, app, caplog, error):
    def fail(_):
        raise error

    monkeypatch.setattr(app, "answer_question", fail)
    with caplog.at_level(logging.ERROR):
        shown = app.ask("question")

    assert shown == app.INDEX_NOT_READY_MESSAGE
    assert "internal" not in shown
    assert str(error) in caplog.text


def test_unexpected_errors_do_not_reach_the_page(monkeypatch, app, caplog):
    def fail(_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app, "answer_question", fail)
    with caplog.at_level(logging.ERROR):
        shown = app.ask("question")

    assert shown == app.GENERIC_ERROR_MESSAGE
    assert "secret internal detail" in caplog.text


def test_empty_question_gets_arabic_prompt(app):
    assert app.ask("   ") == "يرجى إدخال سؤال."


# --- Arabic right-to-left page ---

def test_text_components_are_right_to_left(app):
    text_parts = [c for c in app.demo.blocks.values() if isinstance(c, (gr.Textbox, gr.Markdown))]

    assert len(text_parts) == 3
    assert all(part.rtl for part in text_parts)


def test_page_is_launched_with_right_to_left_css(monkeypatch):
    calls = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: calls.append(kwargs))

    runpy.run_path(str(REPO_ROOT / "app.py"), run_name="__main__")

    assert "direction: rtl" in calls[0]["css"]
    assert calls[0]["footer_links"] == []


def test_citation_id_and_path_are_isolated_from_arabic_text(monkeypatch, app):
    answer = {
        "answer": "answer",
        "sources": [{"id": "S1", "source": "2024/policy.pdf", "page": 3, "distance": 0.1}],
    }
    monkeypatch.setattr(app, "answer_question", lambda question: answer)

    shown = app.ask("question")

    assert "- ⁨[S1]⁩ ⁨2024/policy.pdf⁩ — صفحة 3" in shown
