import os
import tempfile
from pathlib import Path

# Vector store and embedding imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone

# Document processing imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# LLM and chain imports
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

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
    Loads PDF documents from the temporary directory.
    
    Returns:
        documents: List of loaded document objects
    """
    try:
        # TODO: Add validation to check if directory is empty
        loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

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
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    """
    Creates and manages a local vector store using Chroma.
    
    Args:
        texts: List of document chunks
    Returns:
        retriever: Document retriever object
    """
    try:
        # TODO: Add progress indicator for embedding creation
        vectordb = Chroma.from_documents(
            texts, 
            embedding=OpenAIEmbeddings(),
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        vectordb.persist()
        
        # TODO: Experiment with different k values
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        return retriever
    except Exception as e:
        st.error(f"Error creating local vector store: {str(e)}")
        return None

def embeddings_on_pinecone(texts):
    try:
        from pinecone import Pinecone as PineconeClient
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
                    openai_api_key=st.session_state.openai_api_key,
                    model="text-embedding-3-small",
                )

        pc = PineconeClient(api_key=st.session_state.pinecone_api_key)
        vector_store = PineconeVectorStore(
            index=pc.Index(st.session_state.pinecone_index), embedding=embeddings
        )
        vs = vector_store.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
  
        return vs.as_retriever()
    
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {str(e)}")
        return None

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
        # TODO: Add custom prompting for better answers
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=st.secrets.pinecone_api_key),
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
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key"
            )
        
        # TODO: Add validation for API keys
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                help="Enter your Pinecone API key"
            )
        
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input(
                "Pinecone Environment",
                help="Enter your Pinecone environment"
            )
        
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input(
                "Pinecone Index Name",
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
        label="Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )

def process_documents():
    """
    Processes uploaded documents and creates vector store.
    """
    # Validate required fields
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return
    
    if st.session_state.pinecone_db and (
        not st.session_state.pinecone_api_key or
        not st.session_state.pinecone_env or
        not st.session_state.pinecone_index
    ):
        st.warning("Please provide all Pinecone credentials.")
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