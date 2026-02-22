import os
import shutil
from pathlib import Path
import gradio as gr

from Updated_pipeline import (
    query_rag_hist,
    load_docs,
    filter_pages,
    split_docs,
    add_to_chroma,
    DATA_DIR,
    CHAT_HISTORY)

def format_sources(sources):
    formatted = []
    for s in sources:
        try:
            path, page, chunk = s.rsplit(":", 2)
            filename = os.path.basename(path)
            formatted.append(f"• {filename} (page {page})")
        except Exception:
            formatted.append(f"• {s}")
    return "\n".join(formatted)

def chat_fn(message, history):
    response, sources = query_rag_hist(message, CHAT_HISTORY, return_sources=True)
    sources_text = format_sources(sources)
    final_response = f"""{response}
---
### Sources
{sources_text}
"""
    return final_response

def upload_pdf(file):
    if file is None:
        return 'No file Uploaded'

    save_path = Path(DATA_DIR) / Path(file.name).name
    shutil.copy(file.name, save_path)
    docs = load_docs()
    docs = filter_pages(docs)
    chunks = split_docs(docs)
    add_to_chroma(chunks)

    return f"Indexed: {Path(file.name).name}"

def reset_chat_ui():
    CHAT_HISTORY.clear()
    return "", "Chat reset successfully."

def list_documents():
    files = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            files.append(file)

    if not files:
        return "No documents uploaded."

    return "\n".join(f"• {file}" for file in files)

with gr.Blocks() as demo:
    gr.Markdown("# Local RAG PDF QA System")

    with gr.Row():
        with gr.Column(scale = 3):
            chatbot = gr.ChatInterface(fn=chat_fn,
                                       title="Ask questions about your PDFs")
            
            gr.Markdown("### Example Questions")
            gr.Examples(examples = ["What was David Goggins pullup record?",
                                    "How many Hell Weeks did Goggins complete?",
                                    "What is Faster R-CNN?",
                                    "What is ROI pooling used for?",
                                    "What is attention residue?",
                                    "What is shallow work?",
                                    "How does context switching affect cognitive performance?",
                                    ],
                                    inputs = chatbot.textbox)

        with gr.Column(scale=1):
            gr.Markdown("## Documents")
            doc_list = gr.Textbox(value=list_documents(), 
                                  label="Indexed Documents",
                                  interactive=False,
                                  lines=4)
            gr.Markdown("## Upload new PDF")
            file_upload = gr.File(file_types = ['.pdf'],label = 'Select PDF')
            upload_btn = gr.Button("Add Document")
            upload_status = gr.Textbox(label = 'Upload Status')
            
            reset_btn = gr.Button("Reset Chat", variant="secondary")
            reset_status = gr.Markdown()
            
    reset_btn.click(fn=reset_chat_ui, outputs=[chatbot.textbox, reset_status])
    upload_btn.click(fn=upload_pdf, inputs=file_upload, outputs=upload_status).then(fn=list_documents, outputs=doc_list)

demo.launch(inline = False, inbrowser=True)# share=True