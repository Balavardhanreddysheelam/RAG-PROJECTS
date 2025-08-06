# RAG (Retrieval-Augmented Generation) Project

A question-answering system that uses RAG to provide accurate answers from your document collection.

## Features
- Document processing (PDF, TXT)
- Vector store management
- Natural language querying
- Web interface using Streamlit

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `documents` folder and add your documents
4. Run: `streamlit run app.py`

## Environment Variables
Create a `.env` file with:
- CHROMA_PERSIST_DIRECTORY
- EMBEDDING_MODEL
- EMBEDDING_DEVICE

## Usage
1. Initialize the system
2. Upload documents to the `documents` folder
3. Ask questions through the web interface
