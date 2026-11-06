import gradio as gr
from rag_hackathon.pipeline import answer_question

def ask(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."
    try:
        result = answer_question(question)
    except Exception as exc:
        # Keep the demo usable without exposing credentials or internal tracebacks.
        return f"Unable to process the question: {exc}"
    answer = result["answer"]
    if result["sources"]:
        lines = ["\n\n### Sources"]
        for source in result["sources"]:
            lines.append(f"- [{source['id']}] {source['source']} — page {source['page']}")
        answer += "\n" + "\n".join(lines)
    return answer

with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# RAG Assistant\nAsk a question about the indexed documents.")
    question = gr.Textbox(label="Your question", placeholder="Ask a question...", lines=2)
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    submit.click(fn=ask, inputs=question, outputs=answer)
    question.submit(fn=ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
