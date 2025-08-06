import os
import logging
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages vector database operations using HuggingFace embeddings
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "technical_docs"):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = None
        
        # Initialize HuggingFace embeddings (free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print(" Initialized HuggingFace embeddings")
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """Create vector store from documents"""
        if not documents:
            print("✗ No documents provided")
            return False
        
        try:
            print(f" Creating vector store with {len(documents)} documents...")
            print(" This may take a few minutes for large document sets...")
            
            # Create Chroma vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            # Persist the vector store
            self.vector_store.persist()
            
            print(f" Vector store created successfully!")
            return True
            
        except Exception as e:
            print(f" Failed to create vector store: {e}")
            return False
    
    def load_existing_vector_store(self) -> bool:
        """Load existing vector store from disk"""
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                
                # Test if vector store has data
                test_results = self.vector_store.similarity_search("test", k=1)
                print(" Successfully loaded existing vector store")
                return True
            else:
                print(" No existing vector store found")
                return False
                
        except Exception as e:
            print(f" Failed to load existing vector store: {e}")
            return False
    
    def search_similar_documents(self, 
                                query: str, 
                                k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            print(" Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f" Found {len(results)} similar documents for: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f" Search failed: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores"""
        if not self.vector_store:
            print(" Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f" Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            print(f"✗ Search with scores failed: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "HuggingFace"
            }
            
        except Exception as e:
            print(f"✗ Failed to get collection info: {e}")
            return {"error": str(e)}

# Test the vector store
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    print(" Testing Vector Store Manager")
    
    # Initialize components
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    
    # Try to load existing vector store first
    if not vector_manager.load_existing_vector_store():
        print(" Processing documents...")
        documents = processor.process_directory("./documents")
        
        if documents:
            # Create new vector store
            success = vector_manager.create_vector_store(documents)
            
            if success:
                print("🎉 Vector store setup complete!")
        else:
            print(" No documents found in ./documents folder")
            print("Please add PDF files to the documents folder and try again")
    
    # Test search if vector store is ready
    if vector_manager.vector_store:
        print("\n Testing search functionality...")
        test_queries = ["error", "configuration", "troubleshoot"]
        
        for query in test_queries:
            results = vector_manager.search_similar_documents(query, k=2)
            if results:
                print(f"\nQuery: '{query}'")
                for i, doc in enumerate(results[:2]):
                    print(f"  {i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"     Preview: {doc.page_content[:100]}...")
        
        # Show collection info
        info = vector_manager.get_collection_info()
        print(f"\n Collection Info: {info}")
