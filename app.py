import logging
import gradio as gr
from rag_hackathon.generation import MAX_HISTORY_TURNS
from rag_hackathon.pipeline import answer_question
from rag_hackathon.vector_store import EmbeddingModelMismatchError, InterruptedRebuildError
import json
import time
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path("evaluation/results/live_tests.jsonl")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

def save_test(question, answer, response_time, status):
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "response_time_seconds": round(response_time, 3),
        "status": status,
        "correctness": None
    }

    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
logger = logging.getLogger(__name__)

INDEX_NOT_READY_MESSAGE = "فهرس المستندات غير جاهز ويحتاج إلى إعادة بناء من مسؤول النظام."
GENERIC_ERROR_MESSAGE = "تعذّرت معالجة السؤال حالياً. يرجى المحاولة مرة أخرى."
RTL_CSS = ".gradio-container { direction: rtl; }"

def isolate(text: str) -> str:
    """Keep Latin IDs and file paths in one piece inside a right-to-left line."""
    return f"⁨{text}⁩"

def source_location(source: dict) -> str:
    """A text file has no real pages, so only the article locates a passage in it."""
    parts = []
    if source.get("file_type") != "txt":
        parts.append(f"صفحة {source['page']}")
    if source.get("article"):
        parts.append(f"المادة {source['article']}")
    return "، ".join(parts)

def process(question: str, turns: list[dict]) -> tuple[str, dict | None]:
    """Return the text to show and the turn to remember; failures are not remembered."""
    if not question or not question.strip():
        return "يرجى إدخال سؤال.", None
    try:
        result = answer_question(question, turns)
    except (InterruptedRebuildError, EmbeddingModelMismatchError) as exc:
        logger.error("Index is not ready: %s", exc)
        return INDEX_NOT_READY_MESSAGE, None
    except Exception:
        # Details stay in the server log, away from the user-facing page.
        logger.exception("Question processing failed")
        return GENERIC_ERROR_MESSAGE, None
    answer = result["answer"]
    turn = {"question": result.get("search_question", question), "answer": answer}
    if result["sources"]:
        lines = ["\n\n### المصادر"]
        for source in result["sources"]:
            citation = isolate(f"[{source['id']}]")
            location = source_location(source)
            suffix = f" — {location}" if location else ""
            lines.append(f"- {citation} {isolate(source['source'])}{suffix}")
        answer += "\n" + "\n".join(lines)
    return answer, turn

def ask(question: str, turns: list[dict] | None = None) -> str:
    return process(question, turns or [])[0]

def respond(question, chat, turns):
    if not question or not question.strip():
        return "", chat, turns

    start = time.perf_counter()

    shown, turn = process(question, turns)

    elapsed = time.perf_counter() - start

    status = "success" if turn is not None else "failed"

    save_test(
        question=question,
        answer=shown,
        response_time=elapsed,
        status=status
    )

    chat = chat + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": shown}
    ]

    if turn:
        turns = (turns + [turn])[-MAX_HISTORY_TURNS:]

    return "", chat, turns
with gr.Blocks(title="المرشد الأكاديمي الذكي") as demo:
    gr.Markdown("# المرشد الأكاديمي الذكي\nاسأل عن الأنظمة والخطط الدراسية في الجامعة.", rtl=True)
    chat = gr.Chatbot(label="المحادثة", rtl=True)
    question = gr.Textbox(label="سؤالك", placeholder="اكتب سؤالك هنا...", lines=2, rtl=True)
    submit = gr.Button("اسأل", variant="primary")
    turns = gr.State([])
    submit.click(fn=respond, inputs=[question, chat, turns], outputs=[question, chat, turns])
    question.submit(fn=respond, inputs=[question, chat, turns], outputs=[question, chat, turns])

if __name__ == "__main__":
    demo.launch(css=RTL_CSS, footer_links=[])  # The default footer is English-only.
