import gradio as gr
from rag_hackathon.pipeline import answer_question

def ask(question: str) -> str:
    if not question or not question.strip():
        return "يرجى إدخال سؤال."
    try:
        result = answer_question(question)
    except Exception as exc:
        # Keep the demo usable without exposing credentials or internal tracebacks.
        return f"تعذّرت معالجة السؤال: {exc}"
    answer = result["answer"]
    if result["sources"]:
        lines = ["\n\n### المصادر"]
        for source in result["sources"]:
            lines.append(f"- [{source['id']}] {source['source']} — صفحة {source['page']}")
        answer += "\n" + "\n".join(lines)
    return answer

with gr.Blocks(title="مساعد RAG") as demo:
    gr.Markdown("# مساعد RAG\nاطرح سؤالاً عن المستندات المفهرسة.")
    question = gr.Textbox(label="سؤالك", placeholder="اكتب سؤالك هنا...", lines=2)
    submit = gr.Button("اسأل", variant="primary")
    answer = gr.Markdown(label="الإجابة")
    submit.click(fn=ask, inputs=question, outputs=answer)
    question.submit(fn=ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
