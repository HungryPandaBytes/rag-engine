import os
from pathlib import Path
import tempfile
from typing import List
import time

# Vector store and embedding imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

# Document processing imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# LLM and chain imports
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from pinecone import ServerlessSpec,Pinecone as PineconeClient

import streamlit as st

# Set up directory structure
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit page configuration
st.set_page_config(page_title="RAG System")
st.title("📚 Document AI Assistant")
st.write("Upload your documents to get AI-powered insights.")

def validate_api_keys() -> bool:
    """Validate required API keys are present"""
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key")
        return False
    if not st.session_state.index_name:
        st.error("Please enter your Pinecone index name")
        return False
    return True

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in TMP_DIR.glob('*'):
            file.unlink()
    except Exception as e:
        st.error(f"Error cleaning up files: {str(e)}")

def load_documents():
    """Load documents from temp directory"""
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', loader_cls=PyPDFLoader)
    try:
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks with overlap"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(texts):
    """Create Pinecone vectorstore"""
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_api_key,
            model="text-embedding-3-small",
        )

        pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
        
        # Create index if it doesn't exist
        if st.session_state.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=st.session_state.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(20)

        vector_store = PineconeVectorStore(
            index=pc.Index(st.session_state.index_name),
            embedding=embeddings
        )
        
        vs = vector_store.from_documents(texts, embeddings, index_name=st.session_state.index_name)
        return vs.as_retriever()

    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def query_llm(retriever, query):
    """Process queries using the retrieval chain"""
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                openai_api_key=st.secrets['OPENAI_API_KEY'],
                temperature=0.7,
                model="gpt-4-turbo-preview"
            ),
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({
            'question': query, 
            'chat_history': st.session_state.messages
        })
        
        st.session_state.messages.append((query, result['answer']))
        return result['answer']
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "Error processing your question. Please try again."

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Sidebar configuration
    with st.sidebar:
        st.session_state.openai_api_key = st.secrets.get('OPENAI_API_KEY') or st.text_input(
            "OpenAI API Key", type="password"
        )
        st.session_state.index_name = st.secrets.get('PINECONE_INDEX') or st.text_input(
            "Pinecone Index Name"
        )
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Validate keys before processing
    if not validate_api_keys():
        return

    # File upload
    st.session_state.uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Process documents button
    if st.button("Process Documents"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload documents first.")
            return
            
        with st.spinner("Processing documents..."):
            # Save uploaded files
            for uploaded_file in st.session_state.uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=TMP_DIR.as_posix(),
                    suffix='.pdf'
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
            
            # Process documents
            documents = load_documents()
            texts = split_documents(documents)
            st.session_state.retriever = create_vectorstore(texts)
            
            # Cleanup temp files
            cleanup_temp_files()
                
        st.success("Documents processed successfully!")

    # Display chat interface
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])

    # Chat input
    if query := st.chat_input():
        if "retriever" not in st.session_state or st.session_state.retriever is None:
            st.warning("Please process documents first.")
            return
            
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("assistant").write(response)

if __name__ == '__main__':
    main()