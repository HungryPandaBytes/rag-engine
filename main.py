import os
import tempfile
from pathlib import Path

# Vector store and embedding imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Document processing imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
st.title("⚖️ Legal Document AI Assistant")
st.write("Upload your legal documents to get AI-powered analysis, risk assessment, and key insights.")


def load_documents():
    """
    Loads PDF documents using the glob pattern from the temp directory.
    
    Returns:
        documents: List of loaded document objects
    """
    # Define supported file types and their loaders
    supported_types = {
        '.pdf': DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', loader_cls=PyPDFLoader),
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
    text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=0)
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
        from pinecone import Pinecone as PineconeClient, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai.embeddings import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
                    openai_api_key=st.session_state.openai_api_key,
                    model="text-embedding-3-small",
                )

        pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])

        # Check if index exists
        if st.session_state.pinecone_index not in pc.list_indexes().names():
            # Create a new index
            pc.create_index(
                name=st.secrets["PINECONE_INDEX"],          # Your index name
                dimension=1536,             # For OpenAI embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",            # Cloud provider
                    region="us-east-1"      # Region
                )            
            )
        
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
        # Define the system prompt for legal analysis
        system_template = """You are a legal document analyzer with expertise in contract review and risk assessment. When analyzing documents:

        1. Always structure your analysis with clear sections
        2. Highlight potential risks and obligations
        3. Reference specific sections of the source documents
        4. Flag any ambiguous or concerning language
        5. Identify key dates, parties, and monetary terms
        6. Note any missing standard clauses or unusual provisions

        Format your responses with:
        - Summary of relevant findings
        - Key risks or concerns
        - Specific recommendations
        - Reference to source document sections

        For each statement you make, you must cite the source document and relevant section. If you make a claim without finding supporting evidence in the provided context, explicitly state that it's based on general knowledge rather than the source documents.

        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Please provide an answer based on the context above. If you cannot find the answer in the context, please state that explicitly. Remember this is not legal advice and users should consult qualified legal counsel."""
        from langchain.prompts import PromptTemplate
        qa_prompt = PromptTemplate(
            input_variables=["context"],
            template=system_template
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=st.secrets['OPENAI_API_KEY']),
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        # Process the query
        result = qa_chain({
            'question': query, 
            'chat_history': st.session_state.messages
        })
        
        # Update conversation history
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
        # Vector store selection
        # st.session_state.pinecone_db = st.toggle(
        #     'Use Pinecone Vector DB',
        #     help="Toggle between local and cloud vector storage"
        # )

        # Use Pinecone in production
        st.session_state.pinecone_db = True

        st.markdown("**Advanced Settings**")

        # Select chunk size
        chunk_size_map = {
            "XSmall": 250,
            "Small": 500,
            "Medium":1000,
            "Large": 3000
        }

        st.write("Set the document chunk size for Pinecone vector indexing")

        chunk_size_option = st.select_slider(
        "",
        options=[
            "XSmall",
            "Small",
            "Medium",
            "Large"
        ]
        )
        st.session_state.chunk_size = chunk_size_map[chunk_size_option]
        st.write(chunk_size_map[chunk_size_option], "characters")

        # API keys and configuration
        if "OPENAI_API_KEY" in st.secrets:
            st.session_state.openai_api_key = st.secrets['OPENAI_API_KEY']
        else:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key"
            )
        
        if "PINECONE_API_KEY" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets['OPENAI_API_KEY']
        else:
            st.session_state.pinecone_api_key = st.text_input(
                "Pinecone API Key", 
                type="password",
                help="Enter your Pinecone API key"
            )

        if "PINECONE_ENV" in st.secrets:
            st.session_state.pinecone_env = st.secrets['PINECONE_ENV']
        else:
            st.session_state.pinecone_env = st.text_input(
                "Pinecone Environment",
                help="Enter your Pinecone environment"
            )
        
        if "PINECONE_INDEX" in st.secrets:
            st.session_state.pinecone_index = st.secrets['PINECONE_INDEX']
        else:
            st.session_state.pinecone_index = st.text_input(
                "Pinecone Index Name",
                help="Enter your Pinecone index name"
            )
 

    # File upload
    # TODO: Add file size validation
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes

    st.session_state.source_docs = st.file_uploader(
        label="Upload Documents",
        type=["pdf", "docx", "txt", "csv", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents. Supported formats: PDF, Word, Text, CSV, Markdown, Excel"
    )

def validate_prerequisites():
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return False
        
    if st.session_state.pinecone_db:
        if not all([
            st.session_state.openai_api_key,
            st.session_state.pinecone_env, 
            st.session_state.pinecone_index
        ]):
            st.warning("Please provide all Pinecone credentials.")
            return False
            
    if not st.session_state.source_docs:
        st.warning("Please upload at least one document.")
        return False
        
    return True

def save_document_to_temp(doc):
    file_extension = Path(doc.name).suffix
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=TMP_DIR.as_posix(),
        suffix=file_extension
    ) as tmp_file:
        tmp_file.write(doc.read())
        return tmp_file.name

def process_documents():
    """
    Processes uploaded documents and creates vector store.
    """
    if not validate_prerequisites():
        return

    temp_files = []
    try:
        with st.spinner("Processing documents..."):
            # Save uploaded files
            for doc in st.session_state.source_docs:
                temp_file = save_document_to_temp(doc)
                temp_files.append(temp_file)

            # Process documents
            documents = load_documents()
            texts = split_documents(documents)
            
            # Create vector store
            st.session_state.retriever = (
                embeddings_on_pinecone(texts) 
                if st.session_state.pinecone_db
                else embeddings_on_local_vectordb(texts)
            )

        st.success("Documents processed successfully!")

        # Generate response after a sucessful upload to Pinecone
        query_llm(st.session_state.retriever)

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
    finally:
        # Clean up temp files
        for file in temp_files:
            Path(file).unlink(missing_ok=True)

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