r"""
RAG Pipeline with Cross-Encoder Re-Ranking

Features:
- Chroma vector database
- BGE embeddings
- Cross-encoder reranking
- Llama 3.1 via Ollama
- Stateless and conversational modes
- CLI interface

Author: Archit Yadav

Runnable pipeline script converted from Rag_pdf.ipynb .

Usage:
cd to project root: cd "C:/Users/Archit/Documents/ML Projects/RAG-Based-PDF-QA-System" , then: 

  Ingest PDFs into Chroma vector database
    python Code/Updated_pipleine.py ingest

  Ask a single question (stateless)
    python Code/Updated_pipleine.py query --q "What was David Goggins max weight?"

  Start conversational chat mode
    python Code/Updated_pipleine.py chat

  Reset chat history
    python Code/Updated_pipleine.py reset-chat

  Show document and chunk statistics
    python Code/Updated_pipleine.py info

Pipeline stages:
  Ingest:
    PDF → Split → Embed → Store in ChromaDB
  Query:
    Query → Embed → Retrieve Top-10 → Cross-Encoder Re-Rank → Top-4 → LLM Answer
  Chat:
    Query → Rewrite → Retrieve Top-10 → Cross-Encoder Re-Rank → Top-4 → LLM Answer → Save History
"""
import torch
import argparse
import logging
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

# removing loggings for cleaner output, can re-enable if needed for debugging 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Paths and global variables
ROOT = Path(__file__).resolve().parents[1] #Path('C:/Users/Archit/Documents/ML Projects/RAG-Based-PDF-QA-System')
DATA_DIR = ROOT / "Data"
CHROMA_DIR = ROOT / "Chroma"
CHAT_HISTORY = []

# Reset chat history function, can be called from CLI or UI
def reset_chat_history():
    CHAT_HISTORY.clear()
    print("Chat history reset.")

# Load PDFs from the Data directory
def load_docs():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    return loader.load()

# Filter out pages that are too short or contain blacklisted phrases (like copyright notices, TOC, etc)
def filter_pages(docs, min_chars: int = 200):
    cleaned = []
    blacklist = [
        "all rights reserved",
        "copyright",
        "isbn",
        "table of contents",]
    for d in docs:
        text = d.page_content.lower()
        if len(text) < min_chars:
            continue
        if any(b in text for b in blacklist):
            continue
        cleaned.append(d)
    return cleaned

# Split documents into chunks with overlap, and filter out very short chunks
def split_docs(docs: List):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,)
    chunks = text_splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content) > 200]
    return chunks

# The main function to add documents to Chroma, with duplicate checking based on chunk ID
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

# Get embeddings function with device selection
def get_embeddings_function(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings

# Add new chunks to Chroma, avoiding duplicates based on chunk ID
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

# Model Initialization moved to global scope to avoid re-initialization on every query, which was causing latency issues.
emb_fxn = get_embeddings_function()
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=emb_fxn
)
llm = OllamaLLM(model="llama3.1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reranker function 
def rerank(query, docs, top_n=4):
    passages = [doc.page_content for doc, _ in docs]
    pairs = [(query, passage) for passage in passages]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for (doc, _orig_score), _ce_score in scored_docs[:top_n]]


# Simple query without chat history
Base_PROMPT_strict = """
You must answer using ONLY the exact words from the context.

Rules:
- Do NOT explain.
- Do NOT rephrase.
- Do NOT add extra information.
- Return ONLY the answer phrase.

Context:
{context}

Question:
{question}

Answer:
"""

def query_rag(query_text: str, return_context=False):
    results = db.similarity_search_with_score(query_text, k=10)
    reranked_docs = rerank(query_text, results)
    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    prompt_template = ChatPromptTemplate.from_template(Base_PROMPT_strict)
    prompt = prompt_template.format(context=context, question=query_text)
    #print(prompt) 
    response_text = llm.invoke(prompt)
    sources = [doc.metadata.get('id', None) for doc in reranked_docs]
    formatted_response = f'Response: {response_text}\n\nSources: {sources}'
    print(formatted_response)
    if return_context:
        return response_text, context
    return(response_text)

# With chat history 

REWRITE_PROMPT = """
Given the chat history and the latest question, rewrite the question so it is standalone and can be understood without the history.

Chat history:
{history}

Latest question: {question}

Standalone question:
"""

def rewrite_query_with_history(query: str, history: list):
    if not history:
        return query
    history_text = "\n".join(f"{role}: {msg}" for role, msg in history[-6:])
    prompt = REWRITE_PROMPT.format(history=history_text,question=query,)
    #print(prompt)
    rewritten = llm.invoke(prompt)
    return rewritten.strip()

Base_PROMPT_2  = """
Chat history:
{history}

Context:
{context}

Answer the question based on the above context

Question: {question}
"""

def query_rag_hist(query_text: str, history: list, return_context=False, return_sources=False):
    standalone_query = rewrite_query_with_history(query_text, history)
    
    results = db.similarity_search_with_score(standalone_query, k=10)
    reranked_docs = rerank(standalone_query, results)

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    
    history_text = "\n".join(f"{role}: {msg}" for role, msg in history[-6:])
    prompt_template = ChatPromptTemplate.from_template(Base_PROMPT_2)
    prompt = prompt_template.format(context=context, question=query_text, history=history_text,)
    #print(prompt)

    response_text = llm.invoke(prompt)
    history.append(("user", query_text))
    history.append(("assistant", response_text))
    sources = [doc.metadata.get('id', None) for doc in reranked_docs]
    #print("Standalone query:", standalone_query)
    #print("\nResponse: ",response_text)
    #print("\nSources:", sources)
    if return_context and return_sources:
        return response_text, context, sources    
    if return_sources:
        return response_text, sources
    if return_context:
        return response_text, context
    return response_text

def main():
    parser = argparse.ArgumentParser(description="Run PDF -> embeddings -> Chroma pipeline or query RAG system.")
    sub = parser.add_subparsers(dest="cmd")

    ingest = sub.add_parser("ingest", help="Load PDFs, split, and add to Chroma DB")
    #ingest.add_argument("--device", default="cuda", help="device for embeddings model (cpu or cuda)",)

    q = sub.add_parser("query", help="Run a RAG query")
    q.add_argument("--q", required=True, help="Question to ask")

    reset_chat = sub.add_parser("reset-chat", help="Reset the chat history")

    chat = sub.add_parser("chat", help="Run conversational RAG with history")
    #chat.add_argument("--q", required=True, help="Question to ask")
    # interactive chat, no --q arg needed, will prompt user for input

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
    
    elif args.cmd == "chat":
        print("Starting conversational RAG. Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            query_rag_hist(user_input, CHAT_HISTORY)

    elif args.cmd == "reset-chat":
        CHAT_HISTORY.clear()
        logging.info("Chat history reset")

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