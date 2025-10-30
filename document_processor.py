import os
import logging
from typing import List
import pdfplumber
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles PDF document processing, text extraction, and chunking
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    print(f" Successfully extracted text from {os.path.basename(pdf_path)}")
                    return text
        except Exception as e:
            print(f" pdfplumber failed for {pdf_path}: {e}")
        
        try:
            # Fallback to PythonPDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                print(f" Successfully extracted text using PyPDF2 from {os.path.basename(pdf_path)}")
                return text
        except Exception as e:
            print(f" Both extraction methods failed for {pdf_path}: {e}")
            return ""
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all PDF documents from directory"""
        if not os.path.exists(directory_path):
            print(f" Directory {directory_path} does not exist")
            return []
        
        documents = []
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f" No PDF files found in {directory_path}")
            return documents
        
        print(f" Found {len(pdf_files)} PDF files to process")
        
        for filename in pdf_files:
            file_path = os.path.join(directory_path, filename)
            print(f" Processing: {filename}")
            
            text = self.extract_text_from_pdf(file_path)
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "file_path": file_path
                    }
                )
                documents.append(doc)
            else:
                print(f"⚠ No text extracted from {filename}")
        
        print(f" Successfully processed {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        print(f" Chunking {len(documents)} documents...")
        
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata['source']}_chunk_{i}",
                    "chunk_index": i
                })
                chunked_docs.append(chunk)
        
        print(f" Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Complete processing pipeline"""
        print(" Starting document processing...")
        
        # Load documents
        documents = self.load_documents_from_directory(directory_path)
        
        if not documents:
            return []
        
        # Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        return chunked_documents

# Test the processor
if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.process_directory("./documents")
    
    print(f"\n RESULTS:")
    print(f"Total chunks created: {len(docs)}")
    if docs:
        print(f"Sample chunk from {docs[0].metadata['source']}:")
        print(f"Content preview: {docs[0].page_content[:200]}...")
