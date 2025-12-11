import os
import asyncio
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()  # Load environment variables from .env file

async def main():
    # Set up API configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    # base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser=os.getenv("PARSER"),  # Parser selection: mineru or docling
        parse_method=os.getenv("PARSE_METHOD"),  # Parse method: auto, ocr, or txt
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Process a document
    await rag.process_document_complete(
        file_path="inputs/jd_d100_manual.pdf",
        output_dir="./output",
        parse_method=os.getenv("PARSE_METHOD")
    )

    # Query the processed content
    # Pure text query - for basic knowledge base search
    text_result = await rag.aquery(
        "How often do I need to change spark plugs?",
        mode="hybrid"
    )
    print("Text query result:", text_result)

if __name__ == "__main__":
    asyncio.run(main())