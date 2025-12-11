import os
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()

async def init_and_process(device: str):
    """Initialize RAG and process document with specified device."""
    os.environ["DEVICE"] = device
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    config = RAGAnythingConfig(
        working_dir=f"./rag_storage_{device}",
        parser=os.getenv("PARSER", "docling"),
        parse_method=os.getenv("PARSE_METHOD", "txt"),
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
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
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    file_path = "./inputs/condensate_pump.pdf"
    output_dir = f"./output_{device}"
    
    print(f"\n{'='*60}")
    print(f"Processing with DEVICE={device.upper()}")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method=os.getenv("PARSE_METHOD", "txt"),
        )
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.2f} seconds")
        return elapsed
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def main():
    print("RAG Document Processing Benchmark")
    print(f"File: condensate_pump.pdf")
    
    results = {}
    
    # Process on CPU
    cpu_time = await init_and_process("cpu")
    if cpu_time:
        results["cpu"] = cpu_time
    
    # Process on CUDA
    cuda_time = await init_and_process("cuda")
    if cuda_time:
        results["cuda"] = cuda_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    if results:
        for device, elapsed in results.items():
            print(f"{device.upper():<10}: {elapsed:.2f}s")
        
        if len(results) == 2:
            speedup = results["cpu"] / results["cuda"]
            faster = "CUDA" if speedup > 1 else "CPU"
            print(f"\n{faster} is {abs(speedup):.2f}x faster")
    else:
        print("No results to display")

if __name__ == "__main__":
    asyncio.run(main())
