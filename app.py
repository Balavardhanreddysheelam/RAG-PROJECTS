import streamlit as st
import os
import time
import logging
from typing import Dict, Any
from rag import RAGPipeline
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Disable Streamlit warning about script runner
st.set_option('client.showErrorDetails', False)

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'RAG Q&A System - Document Question Answering'
    }
)

# Initialize session state with proper context
if 'script_run_ctx' not in st.session_state:
    st.session_state['script_run_ctx'] = True

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        # Create required directories if they don't exist
        os.makedirs("./documents", exist_ok=True)
        os.makedirs("./chroma_db", exist_ok=True)
        
        with st.spinner("🚀 Initializing RAG system..."):
            if 'rag_pipeline' not in st.session_state or st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline(
                    documents_path=os.path.abspath("./documents"),
                    persist_directory=os.path.abspath("./chroma_db")
                )
            st.session_state.system_ready = True
            st.success("✅ RAG system initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.session_state.system_ready = False
        return False

def display_header():
    """Display the main header"""
    st.title("🤖 RAG Q&A System")
    st.markdown("""
    ### Intelligent Document Query System
    Ask questions about your technical documents using natural language.
    The system will search through your documents and provide accurate answers with source citations.
    """)

def display_sidebar():
    """Display the sidebar with system information and controls"""
    st.sidebar.title("📊 System Status")
    
    # System status
    if st.session_state.system_ready:
        st.sidebar.success("🟢 System Ready")
    else:
        st.sidebar.warning("🟡 System Not Ready")
    
    # Initialize button
    if st.sidebar.button("🔄 Initialize/Restart System"):
        st.session_state.system_ready = False
        st.session_state.rag_pipeline = None
        initialize_rag_system()
    
    st.sidebar.markdown("---")
    
    # System information
    if st.session_state.rag_pipeline:
        st.sidebar.subheader("📋 System Info")
        try:
            system_info = st.session_state.rag_pipeline.get_system_info()
            
            st.sidebar.write(f"**Documents:** {system_info['vector_store'].get('document_count', 0)}")
            st.sidebar.write(f"**Model Status:** {system_info['model_status']}")
            st.sidebar.write(f"**QA Chain:** {system_info['qa_chain_status']}")
            
        except Exception as e:
            st.sidebar.error(f"Error getting system info: {e}")
    
    st.sidebar.markdown("---")
    
    # Document management
    st.sidebar.subheader("📁 Document Management")
    
    if st.sidebar.button("📂 Check Documents"):
        check_documents()
    
    if st.sidebar.button("🔄 Rebuild Vector Store"):
        rebuild_vector_store()
    
    st.sidebar.markdown("---")
    
    # Sample questions
    st.sidebar.subheader("💡 Sample Questions")
    sample_questions = [
        "What are the common error codes?",
        "How do I configure the database?",
        "What should I do for connection errors?",
        "What are the system requirements?",
        "How do I troubleshoot network issues?"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(f"❓ {question}", key=f"sample_{hash(question)}"):
            st.session_state.current_question = question

def check_documents():
    """Check available documents"""
    documents_path = "./documents"
    try:
        if os.path.exists(documents_path):
            files = [f for f in os.listdir(documents_path) if f.lower().endswith(('.pdf', '.txt'))]
            if files:
                st.sidebar.success(f"📚 Found {len(files)} documents:")
                for file in files:
                    st.sidebar.write(f"• {file}")
            else:
                st.sidebar.warning("📭 No documents found in ./documents/")
        else:
            st.sidebar.error("📂 Documents folder not found!")
    except Exception as e:
        st.sidebar.error(f"Error checking documents: {e}")

def rebuild_vector_store():
    """Rebuild the vector store"""
    try:
        with st.spinner("🔄 Rebuilding vector store..."):
            # Delete existing vector store
            if os.path.exists("./chroma_db"):
                import shutil
                shutil.rmtree("./chroma_db")
            
            # Reinitialize system
            st.session_state.system_ready = False
            st.session_state.rag_pipeline = None
            initialize_rag_system()
            
        st.success("✅ Vector store rebuilt successfully!")
    except Exception as e:
        st.error(f"❌ Error rebuilding vector store: {e}")

def display_chat_interface():
    """Display the main chat interface"""
    st.subheader("💬 Ask Your Questions")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g., How do I configure the database connection?",
        key="question_input"
    )
    
    # Clear the current question after displaying
    if "current_question" in st.session_state:
        del st.session_state.current_question
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process question
    if ask_button and question.strip():
        if not st.session_state.system_ready:
            st.error("❌ Please initialize the system first using the sidebar.")
            return
        
        if not st.session_state.rag_pipeline:
            st.error("❌ RAG pipeline not available.")
            return
        
        # Add question to history
        st.session_state.chat_history.append({"type": "question", "content": question, "timestamp": time.time()})
        
        # Get answer
        with st.spinner("🤔 Thinking... Searching through documents..."):
            try:
                result = st.session_state.rag_pipeline.query(question)
                
                # Add answer to history
                st.session_state.chat_history.append({
                    "type": "answer", 
                    "content": result,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
        
        # Clear the input
        st.rerun()

def display_chat_history():
    """Display the chat history"""
    if not st.session_state.chat_history:
        st.info("👋 Welcome! Ask your first question to get started.")
        return
    
    st.subheader("📜 Conversation History")
    
    # Display chat history in reverse order (newest first)
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        if entry["type"] == "question":
            st.markdown(f"**🙋 You asked:** {entry['content']}")
        
        elif entry["type"] == "answer":
            result = entry["content"]
            
            # Display answer
            st.markdown(f"**🤖 Answer:**")
            st.markdown(result["answer"])
            
            # Display confidence and sources
            col1, col2 = st.columns(2)
            
            with col1:
                confidence = result.get("confidence", 0.0)
                if confidence > 0.7:
                    st.success(f"🎯 Confidence: {confidence:.1%}")
                elif confidence > 0.4:
                    st.warning(f"⚠️ Confidence: {confidence:.1%}")
                else:
                    st.error(f"❌ Low Confidence: {confidence:.1%}")
            
            with col2:
                st.info(f"📚 {len(result.get('sources', []))} sources found")
            
            # Display sources
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"📖 View Sources ({len(sources)} documents)"):
                    for j, source in enumerate(sources):
                        st.markdown(f"**Source {j+1}: {source.get('source', 'Unknown')}**")
                        st.markdown(f"Relevance: {source.get('confidence', 0):.1%}")
                        st.markdown(f"Content preview: _{source.get('content_preview', 'No preview')}_")
                        st.markdown("---")
        
        st.markdown("---")

def main():
    """Main application function"""
    try:
        display_header()
        
        # Check if system needs initialization
        if not st.session_state.get("system_ready", False):
            st.warning("🔄 System needs to be initialized. Click 'Initialize/Restart System' in the sidebar.")
            
            # Auto-initialize on first run
            if st.button("🚀 Auto-Initialize System"):
                initialize_rag_system()
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col2:
            display_sidebar()
        
        with col1:
            display_chat_interface()
            st.markdown("---")
            display_chat_history()
            
    except RuntimeError as e:
        if "Missing ScriptRunContext" in str(e):
            st.error("Please run this application using `streamlit run app.py`")
        else:
            raise e

if __name__ == "__main__":
    main()