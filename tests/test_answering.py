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


@pytest.fixture(autouse=True)
def no_real_rewrite(monkeypatch):
    """The pipeline rewrites every question; tests must never reach the real service."""
    monkeypatch.setattr(pipeline, "rewrite_question", lambda question, history: generation.GeneratedAnswer(question))


def make_hit(path, name="policy.pdf", page=3):
    return {
        "text": "evidence",
        "metadata": {"source": name, "source_path": path, "page": page},
        "distance": 0.1,
    }


def use_fake_client(monkeypatch, sent, reply="answer"):
    """Record the request and return a fixed OpenAI-style chat response."""
    class FakeClient:
        def __init__(self, **kwargs):
            sent["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            )

    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    monkeypatch.setattr(generation, "settings", replace(
        config.settings, llm_api_key="test", llm_base_url="https://llm.test/v1", llm_model="test"
    ))


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

    use_fake_client(monkeypatch, sent)

    answer = generation.generate_answer("question", [make_hit("2024/policy.pdf")])

    assert sent["client"]["base_url"] == "https://llm.test/v1"
    assert sent["client"]["api_key"] == "test"
    assert sent["messages"][0] == {"role": "system", "content": generation.SYSTEM_PROMPT}
    assert "Source: 2024/policy.pdf; Page: 3" in sent["messages"][1]["content"]
    assert answer == generation.GeneratedAnswer("answer", 120, 30)


@pytest.mark.parametrize("reply", [None, "", "  \n"])
def test_empty_model_reply_is_an_error(monkeypatch, reply):
    use_fake_client(monkeypatch, {}, reply)

    with pytest.raises(RuntimeError, match="empty answer"):
        generation.generate_answer("question", [make_hit("guide.pdf")])


@pytest.mark.parametrize("name, field", [
    ("LLM_API_KEY", "llm_api_key"),
    ("LLM_BASE_URL", "llm_base_url"),
    ("LLM_MODEL", "llm_model"),
])
def test_missing_llm_setting_is_named_and_no_call_is_made(monkeypatch, name, field):
    sent = {}
    use_fake_client(monkeypatch, sent)
    monkeypatch.setattr(generation, "settings", replace(generation.settings, **{field: ""}))

    with pytest.raises(RuntimeError, match=name):
        generation.generate_answer("question", [make_hit("guide.pdf")])

    assert sent == {}


@pytest.mark.parametrize("variant", [
    "[S1]", "[ S1 ]", "[s1]", "【S1】", "［S1］", "[​S1]", "[‏S1‏]",
    "[س1]", "[س 1]", "[س١]",
])
def test_citation_variants_become_the_standard_form(variant):
    assert generation.normalize_citations(f"الجواب {variant}.") == "الجواب [S1]."


@pytest.mark.parametrize("text", ["[1]", "[المادة 3]", "S1", "45 × 18 = 810", "[S]"])
def test_text_that_is_not_a_citation_is_unchanged(text):
    assert generation.normalize_citations(text) == text


def test_two_digit_citation_keeps_its_number():
    assert generation.normalize_citations("[ S12 ]") == "[S12]"


def test_generated_answer_has_normalized_citations(monkeypatch):
    use_fake_client(monkeypatch, {}, "answer [ S1 ] and 【S2】")

    answer = generation.generate_answer("question", [make_hit("guide.pdf")])

    assert answer.text == "answer [S1] and [S2]"


def test_prompt_names_the_article_when_known(monkeypatch):
    sent = {}
    use_fake_client(monkeypatch, sent)
    hit = make_hit("bylaw.txt", page=1)
    hit["metadata"]["article"] = "12"

    generation.generate_answer("question", [hit])

    assert "Source: bylaw.txt; Page: 1; Article: 12" in sent["messages"][1]["content"]


def test_refusal_without_evidence_needs_no_key_or_model_call(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("no model call is needed")

    monkeypatch.setattr(generation, "OpenAI", forbidden)
    monkeypatch.setattr(
        generation, "settings", replace(config.settings, llm_api_key="", llm_base_url="", llm_model="")
    )

    assert generation.generate_answer("question", []) == generation.GeneratedAnswer(generation.REFUSAL_MESSAGE)


def test_exact_refusal_is_recognized():
    assert generation.GeneratedAnswer(generation.REFUSAL_MESSAGE).refused
    assert generation.GeneratedAnswer(f"**{generation.REFUSAL_MESSAGE}**").refused


def test_answer_mentioning_missing_information_is_not_a_refusal():
    answer = "الرسوم 480 ديناراً [S1]. لا تحتوي الوثائق على معلومات كافية عن رسوم أخرى."

    assert not generation.GeneratedAnswer(answer).refused


def test_refusal_sentence_after_a_real_answer_is_not_a_refusal():
    answer = f"الرسوم 480 ديناراً [S1]. {generation.REFUSAL_MESSAGE}"

    assert not generation.GeneratedAnswer(answer).refused


def test_system_prompt_tells_the_model_the_exact_refusal():
    assert generation.REFUSAL_MESSAGE in generation.SYSTEM_PROMPT


def test_refusal_shows_no_sources(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question: [make_hit("guide.pdf")])
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, found: generation.GeneratedAnswer(generation.REFUSAL_MESSAGE, 50, 9))

    result = pipeline.answer_question("question")

    assert result["refused"] is True
    assert result["sources"] == []
    assert result["retrieved_count"] == 1
    assert result["answer"] == f"{generation.REFUSAL_MESSAGE} {generation.REFERRAL_MESSAGE}"


def test_out_of_scope_reply_is_a_refusal_without_referral(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question: [make_hit("guide.pdf")])
    monkeypatch.setattr(
        pipeline, "generate_answer", lambda question, found: generation.GeneratedAnswer(generation.OUT_OF_SCOPE_MESSAGE)
    )

    result = pipeline.answer_question("question")

    assert result["refused"] is True
    assert result["sources"] == []
    assert result["answer"] == generation.OUT_OF_SCOPE_MESSAGE


def test_system_prompt_ranks_regulations_above_general_pages():
    assert "binding reference" in generation.SYSTEM_PROMPT


def test_system_prompt_tells_the_model_the_out_of_scope_sentence():
    assert generation.OUT_OF_SCOPE_MESSAGE in generation.SYSTEM_PROMPT


def answer_with(monkeypatch, text, paths):
    monkeypatch.setattr(pipeline, "retrieve", lambda question: [make_hit(path) for path in paths])
    monkeypatch.setattr(pipeline, "generate_answer", lambda question, found: generation.GeneratedAnswer(text))
    return pipeline.answer_question("question")


def test_only_cited_sources_are_listed_and_keep_their_ids(monkeypatch):
    result = answer_with(monkeypatch, "answer [S2]", ["a.pdf", "b.pdf", "c.pdf"])

    assert [(s["id"], s["source"]) for s in result["sources"]] == [("S2", "b.pdf")]


def test_uncited_answer_still_lists_what_was_retrieved(monkeypatch):
    result = answer_with(monkeypatch, "answer", ["a.pdf", "b.pdf"])

    assert [s["id"] for s in result["sources"]] == ["S1", "S2"]


def test_citation_outside_the_retrieved_range_is_not_listed(monkeypatch):
    result = answer_with(monkeypatch, "answer [S1] [S9]", ["a.pdf"])

    assert [s["id"] for s in result["sources"]] == ["S1"]


# --- Conversation memory ---

TURN = {"question": "first question", "answer": "first answer"}


def test_first_question_is_rewritten_for_search(monkeypatch):
    seen = {}

    def fake_rewrite(question, history):
        seen["call"] = (question, history)
        return generation.GeneratedAnswer("formal question", 3, 2)

    monkeypatch.setattr(pipeline, "rewrite_question", fake_rewrite)
    result = answer_with(monkeypatch, "answer [S1]", ["a.pdf"])

    assert seen["call"] == ("question", [])
    assert result["search_question"] == "formal question"


def test_rewrite_without_history_marks_the_conversation_empty(monkeypatch):
    sent = {}
    use_fake_client(monkeypatch, sent, "formal question")

    rewritten = generation.rewrite_question("شو بصير؟", [])

    assert rewritten.text == "formal question"
    assert "(none)" in sent["messages"][1]["content"]
    assert "شو بصير؟" in sent["messages"][1]["content"]


def test_source_version_reaches_the_model(monkeypatch):
    sent = {}
    use_fake_client(monkeypatch, sent, "answer [S1]")
    dated = make_hit("a.pdf")
    dated["metadata"]["version"] = "2022/2023"

    generation.generate_answer("question", [dated, make_hit("b.pdf")])

    context = sent["messages"][1]["content"]
    assert "Page: 3; Version: 2022/2023" in context
    assert context.count("Version:") == 1


def test_prompt_prefers_the_newest_source():
    prompt = generation.SYSTEM_PROMPT

    assert "latest Version" in prompt
    assert "no Version" in prompt


def test_follow_up_is_searched_and_answered_as_the_rewritten_question(monkeypatch):
    seen = {}

    def fake_retrieve(question):
        seen["retrieved"] = question
        return [make_hit("a.pdf")]

    def fake_generate(question, found):
        seen["answered"] = question
        return generation.GeneratedAnswer("answer [S1]", 10, 5)

    monkeypatch.setattr(pipeline, "rewrite_question", lambda q, h: generation.GeneratedAnswer("full question", 3, 2))
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "generate_answer", fake_generate)

    result = pipeline.answer_question("and then?", [TURN])

    assert seen == {"retrieved": "full question", "answered": "full question"}
    assert (result["input_tokens"], result["output_tokens"]) == (13, 7)


def test_rewrite_sends_only_recent_turns(monkeypatch):
    sent = {}
    use_fake_client(monkeypatch, sent, "full question")
    history = [{"question": f"q{n}", "answer": f"a{n}"} for n in range(5)]

    rewritten = generation.rewrite_question("and then?", history)

    prompt = sent["messages"][1]["content"]
    assert rewritten.text == "full question"
    assert "q1" not in prompt and "q2" in prompt and "q4" in prompt
    assert "and then?" in prompt


def test_failed_rewrite_falls_back_to_the_question_as_asked(monkeypatch):
    use_fake_client(monkeypatch, {}, None)

    assert generation.rewrite_question("and then?", [TURN]) == generation.GeneratedAnswer("and then?")


def test_chat_remembers_answers_without_the_source_list(monkeypatch, app):
    answer = {
        "answer": "answer [S1]", "search_question": "full question",
        "sources": [{"id": "S1", "source": "a.pdf", "page": 1, "distance": 0.1}],
    }
    monkeypatch.setattr(app, "answer_question", lambda question, history: answer)

    cleared, chat, turns = app.respond("short", [], [])

    assert cleared == ""
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert "a.pdf" in chat[1]["content"]
    assert turns == [{"question": "full question", "answer": "answer [S1]"}]


def test_chat_passes_earlier_turns_and_keeps_only_recent_ones(monkeypatch, app):
    seen = {}

    def fake(question, history):
        seen["history"] = history
        return {"answer": "answer", "sources": []}

    monkeypatch.setattr(app, "answer_question", fake)
    earlier = [{"question": f"q{n}", "answer": "a"} for n in range(3)]

    _, _, turns = app.respond("new", [], earlier)

    assert seen["history"] == earlier
    assert [t["question"] for t in turns] == ["q1", "q2", "new"]


def test_failed_question_is_shown_but_not_remembered(monkeypatch, app):
    def fail(*_):
        raise RuntimeError("down")

    monkeypatch.setattr(app, "answer_question", fail)

    _, chat, turns = app.respond("question", [], [TURN])

    assert chat[1]["content"] == app.GENERIC_ERROR_MESSAGE
    assert turns == [TURN]


def test_blank_message_changes_nothing(app):
    assert app.respond("  ", [], [TURN]) == ("", [], [TURN])


def test_source_label_falls_back_for_older_metadata():
    assert generation.source_label({"source": "policy.pdf"}) == "policy.pdf"
    assert generation.source_label({}) == "Unknown"


def test_page_shows_relative_path_as_citation(monkeypatch, app):
    answer = {
        "answer": "answer",
        "sources": [{"id": "S1", "source": "2024/policy.pdf", "page": 3, "distance": 0.1}],
    }
    monkeypatch.setattr(app, "answer_question", lambda question, history: answer)

    assert "2024/policy.pdf" in app.ask("question")


@pytest.mark.parametrize("error", [
    InterruptedRebuildError("internal rebuild detail"),
    EmbeddingModelMismatchError("internal model detail"),
])
def test_index_errors_show_arabic_message_and_log_details(monkeypatch, app, caplog, error):
    def fail(*_):
        raise error

    monkeypatch.setattr(app, "answer_question", fail)
    with caplog.at_level(logging.ERROR):
        shown = app.ask("question")

    assert shown == app.INDEX_NOT_READY_MESSAGE
    assert "internal" not in shown
    assert str(error) in caplog.text


def test_unexpected_errors_do_not_reach_the_page(monkeypatch, app, caplog):
    def fail(*_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app, "answer_question", fail)
    with caplog.at_level(logging.ERROR):
        shown = app.ask("question")

    assert shown == app.GENERIC_ERROR_MESSAGE
    assert "secret internal detail" in caplog.text


@pytest.mark.parametrize("source, expected", [
    ({"file_type": "txt", "page": 1, "article": "12"}, "المادة 12"),
    ({"file_type": "pdf", "page": 40, "article": "3"}, "صفحة 40، المادة 3"),
    ({"file_type": "pdf", "page": 40, "article": None}, "صفحة 40"),
    ({"file_type": "txt", "page": 1, "article": None}, ""),
])
def test_source_location(app, source, expected):
    assert app.source_location(source) == expected


def test_empty_question_gets_arabic_prompt(app):
    assert app.ask("   ") == "يرجى إدخال سؤال."


# --- Arabic right-to-left page ---

def test_text_components_are_right_to_left(app):
    text_parts = [c for c in app.demo.blocks.values() if isinstance(c, (gr.Textbox, gr.Markdown, gr.Chatbot))]

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
    monkeypatch.setattr(app, "answer_question", lambda question, history: answer)

    shown = app.ask("question")

    assert "- ⁨[S1]⁩ ⁨2024/policy.pdf⁩ — صفحة 3" in shown
