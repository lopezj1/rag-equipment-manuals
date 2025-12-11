import os
import json
import asyncio
from pathlib import Path
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Load environment variables
load_dotenv()

# --- Initialize Session State and RAG Instance ---


@st.cache_resource
def init_rag():
    """Initialize RAGAnything instance (cached for performance)."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser=os.getenv("PARSER", "docling"),
        parse_method=os.getenv("PARSE_METHOD", "txt"),
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-small",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )


def load_processed_files() -> list[str]:
    """Load list of processed files from metadata."""
    metadata_file = Path("./rag_storage/processed_files.json")
    if metadata_file.exists():
        with open(metadata_file) as f:
            data = json.load(f)
            return data.get("files", [])
    return []


def save_processed_files(files: list[str]):
    """Save list of processed files to metadata."""
    metadata_file = Path("./rag_storage/processed_files.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump({"files": files}, f, indent=2)


async def process_uploaded_file(rag, file_path: str, output_dir: str = "./output"):
    """Process a file through the RAG pipeline."""
    try:
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method=os.getenv("PARSE_METHOD", "txt"),
        )
        return True
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False


async def query_rag(rag, user_query: str):
    """Query the RAG instance."""
    try:
        result = await rag.aquery(user_query, mode="hybrid")
        return result
    except Exception as e:
        st.error(f"Error querying RAG: {e}")
        return None


# --- Streamlit UI ---

st.set_page_config(page_title="RAG Equipment Manuals", layout="wide")
st.title("📚 RAG Equipment Manuals")

# Initialize RAG instance
rag = init_rag()

# Sidebar: File Upload
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Upload a PDF to process through the RAG pipeline"
    )

    if uploaded_file is not None:
        # Create inputs directory if it doesn't exist
        os.makedirs("inputs", exist_ok=True)

        # Save uploaded file
        file_path = os.path.join("inputs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            # Process file asynchronously
            success = asyncio.run(
                process_uploaded_file(rag, file_path, output_dir="./output")
            )

            if success:
                # Update processed files list
                processed_files = load_processed_files()
                if uploaded_file.name not in processed_files:
                    processed_files.append(uploaded_file.name)
                    save_processed_files(processed_files)

                st.success(f"✅ Processed: {uploaded_file.name}")
            else:
                st.error("❌ Failed to process file")

# Sidebar: Display Processed Files
with st.sidebar:
    st.header("📋 Processed Files")
    processed_files = load_processed_files()

    if processed_files:
        for idx, file_name in enumerate(processed_files, 1):
            st.write(f"{idx}. {file_name}")
    else:
        st.info("No files processed yet. Upload a PDF to get started!")

# Main Area: Query Interface
st.header("💬 Query Your Documents")
st.write("Ask questions about the processed documents.")

# Chat-like interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # Get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = asyncio.run(query_rag(rag, user_query))

            if response:
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            else:
                st.error("No response generated")

# Footer
st.divider()
st.markdown(
    """
    **RAG Equipment Manuals** | Powered by RAGAnything, OpenAI, and Streamlit
    """
)
