# RAG Equipment Manuals

A Retrieval-Augmented Generation (RAG) system for processing equipment manuals and answering questions about them using LLMs.

## Features

- 📤 **Upload & Process Documents**: Add PDF manuals to the system via a user-friendly Streamlit interface
- 💬 **Interactive Queries**: Ask questions about your documents and get intelligent responses powered by LLMs
- 📋 **Document Tracking**: See a list of all processed files
- 🧠 **RAG Pipeline**: Uses RAGAnything for document parsing and vector-based retrieval

## Setup

### Prerequisites

- Python 3.12+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-equipment-manuals
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Create a `.env` file with your API keys and settings:
```env
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OUTPUT_DIR=./output
PARSER=docling
PARSE_METHOD=txt
```

## Usage

### Streamlit Web Interface (Recommended)

Run the interactive Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then:
1. Upload a PDF in the sidebar under **📤 Upload Documents**
2. Wait for processing to complete
3. Ask questions in the main area under **💬 Query Your Documents**
4. View processed files in the sidebar under **📋 Processed Files**

### Command Line

For one-shot processing and querying:

```bash
python main.py
```

For interactive service mode:

```bash
python main.py --service
```

Service mode commands:
- `process <file_path> [output_dir]` - Process a document
- `query <text>` - Query the RAG system
- `help` - Show available commands
- `exit` - Exit the service

## Project Structure

```
.
├── streamlit_app.py          # Web UI for interactive use
├── main.py                   # CLI entry point
├── document_processing.py    # Document processing logic
├── query_handling.py         # Query execution logic
├── service.py                # Interactive service mode
├── inputs/                   # Upload directory for input PDFs
├── output/                   # Processed document outputs
├── rag_storage/              # Vector store and metadata
└── pyproject.toml            # Project dependencies
```

## Dependencies

- `raganything` - RAG framework
- `docling` / `mineru` - Document parsing
- `openai` - LLM API
- `streamlit` - Web interface
- `python-dotenv` - Environment configuration

## License

MIT