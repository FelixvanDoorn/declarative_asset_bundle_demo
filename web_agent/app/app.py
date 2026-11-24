"""Minimal web UI for the agent: a question box, an Ask button and an answer section."""

import os

import gradio as gr
from agent import ask


def answer(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "_Type a question first._"
    try:
        return ask(question)
    except Exception as e:  # noqa: BLE001 - show any error in the UI instead of a blank answer
        return f"**Error:** {e}"


with gr.Blocks(title="Agent demo") as demo:
    gr.Markdown("# Agent demo\nAsk a question about the Wikipedia articles in the index, e.g. *Where is the Ladin language spoken?*")

    gr.Markdown("## Question")
    question = gr.Textbox(show_label=False, placeholder="Type your question…", lines=2)
    ask_button = gr.Button("Ask", variant="primary")

    gr.Markdown("## Answer")
    answer_box = gr.Markdown("_The answer will appear here._")

    ask_button.click(answer, inputs=question, outputs=answer_box)
    question.submit(answer, inputs=question, outputs=answer_box)


if __name__ == "__main__":
    # Databricks Apps sets DATABRICKS_APP_PORT; 8000 is used locally.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
