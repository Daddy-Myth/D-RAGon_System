"""Runnable pipeline script converted from Final_pipeline.ipynb.ipynb.

Usage:
  python pipeline.py ingest      # load PDFs, split, and add new chunks to Chroma DB
  python pipeline.py query --q "your question"   # run a RAG query
"""
from pathlib import Path
import argparse
import logging
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
CHROMA_DIR = ROOT / "Chroma"


def load_docs():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    return loader.load()


def filter_pages(docs, min_chars: int = 200):
    cleaned = []
    blacklist = [
        "all rights reserved",
        "copyright",
        "isbn",
        "table of contents",
    ]
    for d in docs:
        text = d.page_content.lower()
        if len(text) < min_chars:
            continue
        if any(b in text for b in blacklist):
            continue
        cleaned.append(d)
    return cleaned


def split_docs(docs: List):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content) > 200]
    return chunks


def calc_chunk_ids(chunks: List):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks


def get_embeddings_function(device: str = "cuda"):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        #query_instruction="Represent this sentence for searching relevant passages:",
    )
    return embeddings


def add_to_chroma(chunks: List):
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings_function())
    chunks_with_ids = calc_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items.get("ids", []))
    logging.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    if new_chunks:
        logging.info(f"Adding new documents: {len(new_chunks)}")
        new_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
    else:
        logging.info("No new documents to add")


BASE_PROMPT = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    emb_fxn = get_embeddings_function()
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb_fxn)
    results = db.similarity_search_with_score(query_text, k=4)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(BASE_PROMPT)
    prompt = prompt_template.format(context=context, question=query_text)
    logging.info("Invoking LLM with assembled prompt")
    print(prompt)

    model = OllamaLLM(model="llama3.1")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print("=========================================================================================================================================")
    print(formatted_response)
    return response_text


def main():
    parser = argparse.ArgumentParser(description="Run PDF -> embeddings -> Chroma pipeline or query RAG system.")
    sub = parser.add_subparsers(dest="cmd")

    ingest = sub.add_parser("ingest", help="Load PDFs, split, and add to Chroma DB")
    ingest.add_argument("--device", default="cuda", help="device for embeddings model (cpu or cuda)",)

    q = sub.add_parser("query", help="Run a RAG query")
    q.add_argument("--q", required=True, help="Question to ask")

    info = sub.add_parser("info", help="Show counts of documents and chunks")

    args = parser.parse_args()

    if args.cmd == "ingest":
        logging.info(f"Loading PDFs from {DATA_DIR}")
        docs = load_docs()
        docs = filter_pages(docs)
        chunks = split_docs(docs)
        add_to_chroma(chunks)
        logging.info("Ingestion complete")

    elif args.cmd == "query":
        query_rag(args.q)

    elif args.cmd == "info":
        docs = load_docs()
        docs = filter_pages(docs)
        chunks = split_docs(docs)
        print(f"Pages (after filter): {len(docs)}")
        print(f"Chunks: {len(chunks)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()