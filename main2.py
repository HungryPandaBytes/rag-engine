import os
import tempfile
from pathlib import Path

# Vector store and embedding imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
import pinecone

# Chroma imports
from pathlib import Path
from typing import List, Optional

# Document processing imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# LLM and chain imports
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# UI imports
import streamlit as st


# Set up our directory structure
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Set up the Streamlit page
st.set_page_config(page_title="RAG System")
st.title("📚 Document Q&A System")

def load_documents():
    """
    Loads PDF documents using the glob pattern from the temp directory.
    
    Returns:
        documents: List of loaded document objects
    """
    # Define supported file types and their loaders
    supported_types = {
        '.pdf': DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf'),
        '.txt': DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.txt'),
        '.docx': DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.docx'),
        '.doc': DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.doc')
    }
    documents = []
    for file_type, loader in supported_types.items():
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {file_type} files: {str(e)}")
            return []

    if not documents: 
        st.warning("No supported documents were found in the directory.")

    return documents

def split_documents(documents):
    """
    Splits documents into chunks for processing.
    
    Args:
        documents: List of loaded documents
    Returns:
        texts: List of document chunks
    """
    # TODO: Experiment with different chunk sizes and overlap values
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(documents)

    
    return texts

def embeddings_on_local_vectordb(
    texts: List[str],
    chunk_size: int = 100,
    persist_directory: str = "./chroma_db"
) -> Optional[object]:
    
    """
    Creates and manages a local vector store using Chroma with progress tracking.
    
    Args:
        texts: List of document chunks to embed
        chunk_size: Number of texts to process at once for progress tracking
        persist_directory: Directory to persist the vector store
    Returns:
        retriever: Document retriever object or None if error occurs
    """
    if not texts:
        st.error("No texts provided for embedding")
        return None
        
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process in chunks to show progress
        total_chunks = len(texts)
        
        status_text.text("Initializing embeddings...")
        embedding_function = OpenAIEmbeddings()
        
        status_text.text("Creating vector store...")
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        
        status_text.text("Persisting vector store...")
        vectordb.persist()
        
        # Configure retriever with customizable parameters
        retriever = vectordb.as_retriever(
            search_type="similarity",  # or "mmr" for diversity
            search_kwargs={
                'k': 7,
                'score_threshold': 0.5  # adjust based on your needs
            }
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Successfully created vector store with {len(texts)} documents")
        return retriever
        
    except Exception as e:
        st.error(f"Error creating local vector store: {str(e)}")
        
        # Additional error context
        if "api_key" in str(e).lower():
            st.error("Please check your OpenAI API key configuration")
        elif "dimension" in str(e).lower():
            st.error("Embedding dimension mismatch. Check if all texts are properly formatted")
        
        return None

# Example usage:
LOCAL_VECTOR_STORE_DIR = Path("./chroma_db")
LOCAL_VECTOR_STORE_DIR.mkdir(exist_ok=True)


def embeddings_on_pinecone(texts):
    """
    Creates and manages a Pinecone vector store.
    
    Args:
        texts: List of document chunks
    Returns:
        retriever: Document retriever object
    """
    try:
        # Initialize Pinecone
        pinecone.init(
            api_key=st.session_state.pinecone_api_key,
            environment=st.session_state.pinecone_env
        )
        
        # Create embeddings and store in Pinecone
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
        
        # TODO: Add batch processing for large document sets
        vectordb = Pinecone.from_documents(
            texts, 
            embeddings, 
            index_name=st.session_state.pinecone_index
        )
        
        return vectordb.as_retriever()
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {str(e)}")
        return None
    #     # Initialize Pinecone with new pattern
    #     pc = Pinecone(api_key=st.session_state.pinecone_api_key)
        
    #     # Create embeddings
    #     embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
        
    #     # Create or get index
    #     index_name = st.session_state.pinecone_index
    #     dimension = 1536  # OpenAI embeddings dimension
        
    #     # Check if index exists, create if it doesn't
    #     with st.spinner("Initializing Pinecone index..."):
    #         if index_name not in pc.list_indexes().names():
    #             pc.create_index(
    #                 name=index_name,
    #                 dimension=dimension,
    #                 metric='cosine',
                  
    #             )
        
    #     # Create vector store
    #     vectordb = Pinecone.from_documents(
    #         documents=texts,
    #         embedding=embeddings,
    #         index_name=index_name,
    #         client=pc  # Pass the Pinecone client instance
    #     )
        
    #     return vectordb.as_retriever()
    # except Exception as e:
    #     st.error(f"Error creating Pinecone vector store: {str(e)}")
    #     return None

def query_llm(retriever, query):
    """
    Processes queries using the retrieval chain.
    
    Args:
        retriever: Document retriever object
        query: User question string
    Returns:
        result: Generated answer
    """
    try:
        print(retriever)
        # TODO: Add custom prompting for better answers
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=st.session_state.openai_api_key),
            retriever=retriever,
            return_source_documents=True,
        )
        
        # Process the query
        result = qa_chain({
            'question': query, 
            'chat_history': st.session_state.messages
        })
        
        # Update conversation history
        # TODO: Add source citations to the response
        st.session_state.messages.append((query, result['answer']))
        
        return result['answer']
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I encountered an error processing your question. Please try again."

def setup_interface():
    """
    Sets up the Streamlit interface components.
    """
    with st.sidebar:
        # API keys and configuration
        st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
        st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key", 
            disabled=True
        )
        
        # TODO: Add validation for API keys
        st.session_state.pinecone_api_key = st.secrets.get("PINECONE_API_KEY", "")
        st.text_input(
            "Pinecone API Key",
            value=st.session_state.pinecone_api_key,
            type="password",
            help="Enter your Pinecone API key", 
            disabled=True
        )
        
        st.session_state.pinecone_env = st.secrets.get("PINECONE_ENV", "")
        st.session_state.pinecone_env = st.text_input(
            "Pinecone Region",  # Changed from Environment to Region
            value=st.session_state.pinecone_env,
            help="Enter your Pinecone region (e.g., us-west-2)"
        )
        
        st.session_state.pinecone_index = st.secrets.get("PINECONE_INDEX", "")
        st.session_state.pinecone_index = st.text_input(
            "Pinecone Index Name",
            value=st.session_state.pinecone_index,
            help="Enter your Pinecone index name"
        )
    
    # Vector store selection
    st.session_state.pinecone_db = st.toggle(
        'Use Pinecone Vector DB',
        help="Toggle between local and cloud vector storage"
    )
    
    # File upload
    # TODO: Add file size validation
    st.session_state.source_docs = st.file_uploader(
        label="Upload Documents",
        type=["pdf", "docx", "txt", "csv", "md", "xlsx"],
        accept_multiple_files=True,
        help="Upload one or more documents. Supported formats: PDF, Word, Text, CSV, Markdown, Excel"
    )

def process_documents():
    """
    Processes uploaded documents and creates vector store.
    """
    # Validate required fields
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return
    
    if st.session_state.pinecone_db:
        if not st.session_state.pinecone_api_key.strip():
            st.warning("Please enter a valid Pinecone API key.")
            return
        if not st.session_state.pinecone_env.strip():
            st.warning("Please enter a valid Pinecone region.")
            return
        if not st.session_state.pinecone_index.strip():
            st.warning("Please enter a valid Pinecone index name.")
            return
    
    if not st.session_state.source_docs:
        st.warning("Please upload at least one document.")
        return
    
    try:
        with st.spinner("Processing documents..."):
            # Save uploaded files to temporary directory
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    dir=TMP_DIR.as_posix(),
                    suffix='.pdf'
                ) as tmp_file:
                    tmp_file.write(source_doc.read())
            
            # Load and process documents
            documents = load_documents()
            
            # Clean up temporary files
            for file in TMP_DIR.iterdir():
                TMP_DIR.joinpath(file).unlink()
            
            # Split documents into chunks
            texts = split_documents(documents)
            
            # Create vector store
            if not st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
            else:
                st.session_state.retriever = embeddings_on_pinecone(texts)
            
            st.success("Documents processed successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    """
    Main application loop.
    """
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Set up the interface
    setup_interface()
    
    # Process documents button
    st.button("Process Documents", on_click=process_documents)
    
    # Display chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])
    
    # Chat input
    if query := st.chat_input():
        if "retriever" not in st.session_state:
            st.warning("Please process documents first.")
            return
            
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("assistant").write(response)

if __name__ == '__main__':
    main()