# 📘 RAG Project Design Notes

---

## 🔵 Project Info
**Working Title:** LLM-Powered PDF RAG System  
**Short Name:** PDF RAG / LLM PDF RAG  

**Problem Statement:**  
Large collections of private documents such as research papers, reports, and manuals are difficult to search using traditional keyword-based methods. Users often waste time locating relevant information across multiple PDF files, and existing tools do not provide precise, context-aware answers to natural language queries.  

This project aims to design and implement an **LLM-powered Retrieval-Augmented Generation (RAG) system** that enables users to upload private PDF documents and query them using natural language by combining semantic retrieval with generative models.

---

## 🎯 Objectives
- Enable natural-language querying over private PDF document collections  
- Retrieve relevant document passages using vector similarity search  
- Generate grounded answers using an LLM  
- Reduce hallucinations via retrieval and citation  
- Provide a simple web interface for interaction  

---

## 🧠 Phase 1: Core Concepts

### What is RAG?
A system that retrieves relevant document chunks from an external knowledge store and supplies them to a large language model so answers are grounded in source material.

### Why RAG Instead of Fine-Tuning?
Fine-tuning is expensive, static, and unsuitable for frequently changing document collections, whereas RAG allows new documents to be added without retraining the model.

### Key Terms
- **Embeddings:** Vector representations of text used for semantic similarity search  
- **Vector Database:** Stores and indexes embeddings for fast retrieval  
- **Chunking:** Splitting documents into smaller passages before embedding  
- **Retriever:** Component that selects the most relevant chunks for a query  
- **Hallucination:** LLM generating unsupported information  

---

## 🏗️ System Pipeline
PDF Upload → Text Extraction → Chunking → Embeddings → Vector DB → Retrieval → LLM → UI

---

## 🧰 Tools Being Considered

### PDF Ingestion
- Library:
- OCR Tool:
- Metadata Handling:

### Chunking
- Method:
- Chunk Size:
- Overlap:

### Embeddings
- Model:
- Dimensionality:
- Local / API:

### Vector Database
- Tool:
- Persistence:
- Metadata Filters:

### Retriever
- Top-K:
- Similarity Metric:
- Reranker:

### LLM
- Model:
- Local / API:
- Hardware:
- Cost:

### Framework / Orchestration
- LangChain / LlamaIndex / Custom:
- Reason for Choice:

### Backend / API
- Tool:
-

### UI
- Tool:
-

---

## 🧪 Evaluation Plan

### Metrics
- Retrieval recall@k:
- Answer accuracy:
- Hallucination rate:
- Latency:

### Test Dataset
-

---

## 👥 Team Roles

- Lead / Integration:
- PDF Ingestion:
- Retrieval:
- Backend:
- Frontend:

---

## ⚠️ Risks & Open Questions
-
-
-

---

## 📈 Scalability & Future Work
-
-
-

---

## 📝 Notes from Videos
- **Video Title:**
  - Key takeaway:
  - Tools used:
  - Design choices:
