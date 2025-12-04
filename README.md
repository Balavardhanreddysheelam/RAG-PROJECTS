# RAG (Retrieval-Augmented Generation) Question-Answering System

A production-ready question-answering system that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from your custom document collection. This system combines semantic search capabilities with large language models to deliver intelligent responses grounded in your documents.

## 🎯 Overview

This RAG system implements a complete pipeline for document-based question answering:

1. **Document Processing** - Ingest and parse PDFs, text files, and other documents
2. **Vector Embedding** - Convert documents into semantic embeddings for efficient retrieval
3. **Semantic Search** - Retrieve the most relevant document chunks using similarity matching
4. **Answer Generation** - Generate answers using an LLM augmented with retrieved context

The system is designed to be scalable, maintainable, and easy to deploy as a web application.

## ✨ Features

- **Multi-format Document Support**
  - PDF files with automatic text extraction
  - Plain text files
  - Extensible architecture for additional formats

- **Vector Store Management**
  - Persistent vector storage using Chroma
  - Efficient semantic similarity search
  - Batch document processing
  - Vector store optimization and cleanup

- **Natural Language Querying**
  - Ask questions in natural language
  - Retrieves relevant document chunks automatically
  - Provides source citations with retrieved content

- **Web Interface**
  - User-friendly Streamlit-based UI
  - Real-time document upload
  - Interactive Q&A interface
  - Source document visualization

- **Configurable Components**
  - Pluggable embedding models
  - Configurable LLM backends
  - Adjustable retrieval parameters
  - Device selection (CPU/GPU)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Sufficient disk space for vector store (~500MB - 5GB depending on document size)
- GPU recommended (but CPU-only operation supported)
