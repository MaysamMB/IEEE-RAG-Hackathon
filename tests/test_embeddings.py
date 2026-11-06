"""Prefix tests; no model is loaded."""
from dataclasses import replace

from rag_hackathon import config, embeddings
from rag_hackathon.embeddings import add_task_prefix, embed_texts


def use_model(monkeypatch, name):
    monkeypatch.setattr(embeddings, "settings", replace(config.settings, embedding_model=name))


def test_e5_marks_questions_and_passages(monkeypatch):
    use_model(monkeypatch, "intfloat/multilingual-e5-small")

    assert add_task_prefix(["نص"], is_query=True) == ["query: نص"]
    assert add_task_prefix(["نص"], is_query=False) == ["passage: نص"]


def test_other_models_get_plain_text(monkeypatch):
    use_model(monkeypatch, "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    assert add_task_prefix(["نص"], is_query=True) == ["نص"]


def test_documents_are_embedded_as_passages_by_default(monkeypatch):
    use_model(monkeypatch, "intfloat/multilingual-e5-small")
    seen = []

    class FakeModel:
        def encode(self, texts, **_):
            seen.extend(texts)
            return type("Vectors", (), {"tolist": lambda self: [[1.0]]})()

    monkeypatch.setattr(embeddings, "get_embedder", lambda: FakeModel())

    embed_texts(["نص"])

    assert seen == ["passage: نص"]
