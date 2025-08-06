import os
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline
    """
    
    def __init__(self, 
                 documents_path: str = "./documents",
                 persist_directory: str = "./chroma_db",
                 model_name: str = "microsoft/DialoGPT-medium"):
        
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager(persist_directory=persist_directory)
        self.llm = None
        self.qa_chain = None
        
        # Setup RAG pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the complete RAG pipeline"""
        print("🚀 Setting up RAG pipeline...")
        
        # Step 1: Setup vector store
        if not self._setup_vector_store():
            return False
        
        # Step 2: Initialize LLM
        if not self._initialize_llm():
            return False
        
        # Step 3: Create QA chain
        self._create_qa_chain()
        
        print("✅ RAG pipeline setup complete!")
        return True
    
    def _setup_vector_store(self) -> bool:
        """Setup vector store with documents"""
        try:
            # Try to load existing vector store first
            if self.vector_manager.load_existing_vector_store():
                print("📚 Using existing vector store")
                return True
            
            # If no existing store, process documents and create new one
            print("📁 Processing documents...")
            documents = self.document_processor.process_directory(self.documents_path)
            
            if not documents:
                print("❌ No documents found to process!")
                return False
            
            # Create vector store
            success = self.vector_manager.create_vector_store(documents)
            return success
            
        except Exception as e:
            print(f"❌ Failed to setup vector store: {e}")
            return False
    
    def _initialize_llm(self) -> bool:
        """Initialize the language model"""
        try:
            print("🤖 Initializing language model...")
            print("⏳ This may take a few minutes on first run...")
            
            # Use a smaller, faster model for better performance
            model_name = "microsoft/DialoGPT-small"  # Smaller model for faster inference
            
            # Create text generation pipeline
            text_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device=-1  # Use CPU
            )
            
            # Create HuggingFace LLM
            self.llm = HuggingFacePipeline(pipeline=text_pipeline)
            
            print("✅ Language model initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            print("🔄 Falling back to simple text generation...")
            
            # Fallback to a simple response generator
            self.llm = self._create_fallback_llm()
            return True
    
    def _create_fallback_llm(self):
        """Create a simple fallback LLM"""
        class SimpleLLM:
            def __call__(self, prompt):
                # Simple rule-based responses
                prompt_lower = prompt.lower()
                
                if "error" in prompt_lower or "problem" in prompt_lower:
                    return "Based on the documentation, here are the troubleshooting steps you should follow..."
                elif "configure" in prompt_lower or "setup" in prompt_lower:
                    return "According to the configuration guide, here's how to set this up..."
                elif "install" in prompt_lower:
                    return "For installation procedures, please follow these steps..."
                else:
                    return "Based on the available documentation, here's the relevant information..."
        
        return SimpleLLM()
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        try:
            # Create custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context, just say "I don't have enough information in the provided documents to answer this question."

            Context: {context}

            Question: {question}

            Answer: """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_manager.vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv("TOP_K_RESULTS", 5))}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            print("✅ QA chain created successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to create advanced QA chain: {e}")
            print("🔄 Using simple retrieval system...")
            self.qa_chain = None
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and return an answer with sources
        """
        try:
            if not self.vector_manager.vector_store:
                return {
                    "answer": "❌ Vector store not available. Please check if documents are processed correctly.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Get relevant documents
            relevant_docs = self.vector_manager.search_with_scores(question, k=5)
            
            if not relevant_docs:
                return {
                    "answer": "❌ No relevant information found in the documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract context from relevant documents
            context = "\n\n".join([doc.page_content for doc, score in relevant_docs])
            
            # Generate answer using simple approach
            answer = self._generate_answer(question, context, relevant_docs)
            
            # Extract source information
            sources = []
            for doc, score in relevant_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:200] + "...",
                    "confidence": round(1 - score, 2)  # Convert distance to confidence
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(sum(1 - score for doc, score in relevant_docs) / len(relevant_docs), 2)
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"❌ Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_answer(self, question: str, context: str, relevant_docs: List) -> str:
        """Generate answer based on context"""
        try:
            if self.qa_chain:
                # Use advanced QA chain if available
                result = self.qa_chain({"query": question})
                return result["result"]
            else:
                # Use simple context-based answering
                return self._simple_answer_generation(question, context, relevant_docs)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._simple_answer_generation(question, context, relevant_docs)
    
    def _simple_answer_generation(self, question: str, context: str, relevant_docs: List) -> str:
        """Simple context-based answer generation"""
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Extract key information based on question type
        if "error" in question_lower or "problem" in question_lower or "issue" in question_lower:
            if "error code" in context_lower:
                answer = "Based on the documentation, here are the error codes and solutions I found:\n\n"
                # Extract error code information
                lines = context.split('\n')
                for line in lines:
                    if "error code" in line.lower() or "error" in line.lower():
                        answer += f"• {line.strip()}\n"
                return answer
            else:
                return "Based on the troubleshooting documentation, here are the relevant solutions:\n\n" + context[:500] + "..."
        
        elif "configure" in question_lower or "setup" in question_lower or "install" in question_lower:
            return "According to the configuration guide:\n\n" + context[:500] + "..."
        
        elif "how" in question_lower:
            return "Here's how to do this based on the documentation:\n\n" + context[:500] + "..."
        
        else:
            # General answer
            return "Based on the available documentation:\n\n" + context[:500] + "..."
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system"""
        vector_info = self.vector_manager.get_collection_info()
        
        return {
            "vector_store": vector_info,
            "documents_path": self.documents_path,
            "model_status": "Loaded" if self.llm else "Not loaded",
            "qa_chain_status": "Available" if self.qa_chain else "Simple mode"
        }

# Test the RAG pipeline
if __name__ == "__main__":
    print("🧪 Testing RAG Pipeline...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test queries
    test_questions = [
             "How does the MySQL connection work in the pipeline?",
             "What is the structure of the capstone project pipeline?",
             "How are the Python assignments implemented?",
             "What are the key components of the pipeline implementation?"
]
    
    print("\n🔍 Testing queries...")
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = rag.query(question)
        print(f"💡 Answer: {result['answer'][:200]}...")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"📚 Sources: {len(result['sources'])} documents found")
    
    # System info
    info = rag.get_system_info()
    print(f"\n📊 System Info: {info}")