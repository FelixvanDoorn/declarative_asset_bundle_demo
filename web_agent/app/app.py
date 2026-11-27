"""Minimal web UI for the agent: a question box, an Ask button and an answer section."""

import os

import gradio as gr
from agent import ask


LOADING = "⏳ _Searching Wikipedia and thinking… this usually takes a few seconds._"


def answer(question: str):
    """Show a loading message and a disabled button right away, then the answer."""
    question = (question or "").strip()
    if not question:
        yield "_Type a question first._", gr.Button("Ask", interactive=True)
        return
    yield LOADING, gr.Button("Thinking…", interactive=False)
    try:
        result = ask(question)
    except Exception as e:  # noqa: BLE001 - show any error in the UI instead of a blank answer
        result = f"**Error:** {e}"
    yield result, gr.Button("Ask", interactive=True)


with gr.Blocks(title="Agent demo") as demo:
    gr.Markdown("# Agent demo\nAsk a question about the Wikipedia articles in the index, e.g. *Where is the Ladin language spoken?*")

    gr.Markdown("## Question")
    question = gr.Textbox(show_label=False, placeholder="Type your question…", lines=2)
    ask_button = gr.Button("Ask", variant="primary")

    gr.Markdown("## Answer")
    answer_box = gr.Markdown("_The answer will appear here._")

    # show_progress="hidden": our own loading message replaces Gradio's default overlay.
    ask_button.click(answer, inputs=question, outputs=[answer_box, ask_button], show_progress="hidden")
    question.submit(answer, inputs=question, outputs=[answer_box, ask_button], show_progress="hidden")


if __name__ == "__main__":
    # Databricks Apps sets DATABRICKS_APP_PORT; 8000 is used locally.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
