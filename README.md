# Multimodal RAG for PDFs with Images

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about PDF documents containing both text and images. Text and images are embedded in a shared vector space using CLIP, retrieved together with a single query, and passed to a vision-capable language model for grounded answers.

## Overview

Traditional RAG systems handle text well but often ignore charts, diagrams, and other visuals in PDFs. This project treats text chunks and extracted images as first-class retrieval targets:

1. **Ingest** a PDF and extract text and embedded images page by page.
2. **Embed** both modalities with CLIP so they live in the same embedding space.
3. **Index** all chunks in a FAISS vector store.
4. **Retrieve** the most relevant text and images for a natural-language query.
5. **Generate** an answer with GPT-4.1, sending retrieved text and base64-encoded images as multimodal context.

The included notebook (`1-multimodalopenai.ipynb`) walks through the full pipeline end to end.

## Architecture

```
PDF Document
    |
    +-- Text extraction (PyMuPDF)
    |       |
    |       +-- Chunking (RecursiveCharacterTextSplitter)
    |       +-- CLIP text embeddings
    |
    +-- Image extraction (PyMuPDF)
            |
            +-- PIL conversion + base64 storage (for LLM)
            +-- CLIP image embeddings
                    |
                    v
            Unified FAISS vector store
                    |
            Query -> CLIP text embedding -> similarity search
                    |
                    v
            GPT-4.1 (text + image context) -> Answer
```

## Features

- Unified retrieval across text and images using CLIP (`openai/clip-vit-base-patch32`)
- PDF parsing with PyMuPDF for text and embedded raster images
- Vector search with LangChain and FAISS
- Multimodal prompting with LangChain and OpenAI GPT-4.1
- Configurable chunk size and retrieval depth (`k`)

## Prerequisites

- Python 3.10 or newer (3.11 recommended)
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to GPT-4.1 (or update the model name in the notebook)
- Enough disk space for PyTorch and the CLIP model weights (downloaded on first run)
- Optional: CUDA-capable GPU for faster CLIP inference (CPU works but is slower)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gayuth-W/Multimodel_RAG.git
cd Multimodel_RAG
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install pymupdf pillow torch transformers numpy scikit-learn python-dotenv
pip install langchain langchain-core langchain-community langchain-openai faiss-cpu
pip install jupyter ipywidgets
```

Use `faiss-gpu` instead of `faiss-cpu` if you have a compatible NVIDIA GPU and want GPU-accelerated search.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The notebook loads this key via `python-dotenv` and sets `OPENAI_API_KEY` for LangChain.

### 5. Add a PDF to process

Place your target PDF in the project root. The notebook defaults to:

```
multimodal_sample.pdf
```

A sample PDF is used locally for development but is listed in `.gitignore` and is not committed to the repository. Use any PDF that contains both text and images (reports, slides, dashboards, and so on).

## Usage

### Run the notebook

```bash
jupyter notebook 1-multimodalopenai.ipynb
```

Execute cells in order. The notebook will:

1. Load the CLIP model and processor
2. Parse the PDF and build text chunks plus image documents
3. Create a FAISS index from precomputed CLIP embeddings
4. Initialize the OpenAI chat model
5. Run example queries through the RAG pipeline

### Example queries

```python
queries = [
    "What does the chart on page 1 show about revenue trends?",
    "Summarize the main findings from the document",
    "What visual elements are present in the document?",
]
```

### Programmatic flow

The core pipeline is exposed as `multimodal_pdf_rag_pipeline(query)`:

```python
answer = multimodal_pdf_rag_pipeline("What does the chart show about Q3 revenue?")
print(answer)
```

Under the hood:

| Function | Role |
|----------|------|
| `embed_text(text)` | CLIP text embedding (L2-normalized) |
| `embed_image(image)` | CLIP image embedding from a path or PIL image |
| `retrieve_multimodal(query, k=5)` | Embed the query and search FAISS |
| `create_multimodal_message(query, docs)` | Build a `HumanMessage` with text excerpts and base64 images |
| `multimodal_pdf_rag_pipeline(query)` | Retrieve, prompt, and return the model response |

## Project structure

```
multimodal_rag/
├── 1-multimodalopenai.ipynb   # Main notebook: ingestion, indexing, and RAG
├── multimodal_sample.pdf      # Local sample PDF (gitignored; not in repo)
├── .gitignore
└── README.md
```

## How it works (detail)

### PDF ingestion

- **Text:** Each page's text is split with `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=100`). Each chunk is embedded with CLIP and stored as a LangChain `Document` with `metadata={"page": i, "type": "text"}`.
- **Images:** Embedded images are extracted per page, converted to RGB PIL images, and stored as base64 PNG strings keyed by `page_{i}_img_{j}`. Each image gets a CLIP embedding and a placeholder document (`page_content="[Image: ...]"`) with `metadata={"type": "image", "image_id": ...}`.

### Vector store

Embeddings are stacked into a NumPy array and loaded into FAISS with `FAISS.from_embeddings`, pairing each embedding with its document content and metadata.

### Retrieval and generation

The user query is embedded with the same CLIP text encoder used at index time. Top-`k` results can include both text and images. Retrieved images are injected into the prompt as `image_url` content blocks (`data:image/png;base64,...`) alongside text excerpts, then sent to GPT-4.1.

## Configuration

You can tune behavior by editing values in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pdf_path` | `multimodal_sample.pdf` | Input PDF file |
| `chunk_size` | `500` | Characters per text chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |
| `k` | `5` | Number of documents retrieved per query |
| CLIP model | `openai/clip-vit-base-patch32` | Shared embedding model |
| LLM | `openai:gpt-4.1` | Chat model for answer generation |

## Dependencies

| Package | Purpose |
|---------|---------|
| PyMuPDF (`fitz`) | PDF text and image extraction |
| Pillow | Image handling |
| PyTorch | CLIP inference |
| Transformers | CLIP model and processor |
| NumPy | Embedding arrays |
| scikit-learn | Cosine similarity utilities (imported; optional for extensions) |
| LangChain | Documents, splitting, vector store, chat model, messages |
| FAISS | Approximate nearest-neighbor search |
| python-dotenv | Load `OPENAI_API_KEY` from `.env` |
| Jupyter | Run the notebook |

## Limitations

- **Embedded images only:** PyMuPDF extracts raster images embedded in the PDF. Text rendered as vector graphics or screenshots flattened into a page may not be recovered as separate images.
- **CLIP retrieval quality:** CLIP aligns text and images in a general-purpose space; domain-specific or fine-grained document QA may need a different embedder or hybrid retrieval.
- **Memory:** All image base64 payloads are held in `image_data_store` in memory for the session. Large PDFs with many high-resolution images can be heavy.
- **API cost:** Each query invokes the OpenAI API; multimodal prompts with images use more tokens than text-only RAG.
- **Single PDF per run:** The notebook indexes one PDF at a time. Multi-document support would require extending metadata and the ingestion loop.

## Troubleshooting

**`OPENAI_API_KEY` not set**

Ensure `.env` exists in the project root and contains a valid key. Restart the Jupyter kernel after creating or editing `.env`.

**`multimodal_sample.pdf` not found**

Add your own PDF or change `pdf_path` in the notebook to point to an existing file.

**Slow first run**

CLIP weights are downloaded from Hugging Face on first use. Subsequent runs reuse the cached model.

**FAISS warning about `embedding_function`**

LangChain may log a deprecation notice when passing precomputed embeddings. The notebook still works; you can migrate to a custom `Embeddings` wrapper if you refactor to a script.

**Jupyter widget warning**

If you see a `tqdm` / `IProgress` warning, install widgets:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## License

No license file is included in this repository. Add one if you plan to distribute or open-source the project formally.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for unified text-image embeddings
- [LangChain](https://github.com/langchain-ai/langchain) for RAG orchestration
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing
