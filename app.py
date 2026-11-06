import logging
import gradio as gr
from rag_hackathon.pipeline import answer_question
from rag_hackathon.vector_store import EmbeddingModelMismatchError, InterruptedRebuildError

logger = logging.getLogger(__name__)

INDEX_NOT_READY_MESSAGE = "فهرس المستندات غير جاهز ويحتاج إلى إعادة بناء من مسؤول النظام."
GENERIC_ERROR_MESSAGE = "تعذّرت معالجة السؤال حالياً. يرجى المحاولة مرة أخرى."
RTL_CSS = ".gradio-container { direction: rtl; }"

def isolate(text: str) -> str:
    """Keep Latin IDs and file paths in one piece inside a right-to-left line."""
    return f"⁨{text}⁩"

def ask(question: str) -> str:
    if not question or not question.strip():
        return "يرجى إدخال سؤال."
    try:
        result = answer_question(question)
    except (InterruptedRebuildError, EmbeddingModelMismatchError) as exc:
        logger.error("Index is not ready: %s", exc)
        return INDEX_NOT_READY_MESSAGE
    except Exception:
        # Details stay in the server log, away from the user-facing page.
        logger.exception("Question processing failed")
        return GENERIC_ERROR_MESSAGE
    answer = result["answer"]
    if result["sources"]:
        lines = ["\n\n### المصادر"]
        for source in result["sources"]:
            citation = isolate(f"[{source['id']}]")
            lines.append(f"- {citation} {isolate(source['source'])} — صفحة {source['page']}")
        answer += "\n" + "\n".join(lines)
    return answer

with gr.Blocks(title="مساعد RAG") as demo:
    gr.Markdown("# مساعد RAG\nاطرح سؤالاً عن المستندات المفهرسة.", rtl=True)
    question = gr.Textbox(label="سؤالك", placeholder="اكتب سؤالك هنا...", lines=2, rtl=True)
    submit = gr.Button("اسأل", variant="primary")
    answer = gr.Markdown(label="الإجابة", rtl=True)
    submit.click(fn=ask, inputs=question, outputs=answer)
    question.submit(fn=ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(css=RTL_CSS, footer_links=[])  # The default footer is English-only.
