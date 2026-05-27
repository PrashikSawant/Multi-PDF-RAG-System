# 📑 Day 13 — Multi-PDF RAG System

Upload multiple PDFs and TXT files, ask questions across 
all of them, or compare documents side by side.
Full RAG pipeline with multi-document semantic search.

## 💡 What It Does
- Upload multiple PDFs and TXT files simultaneously
- Ask questions across all documents at once
- Filter search to specific documents using multiselect
- Compare mode — AI compares documents side by side
- Chunk-level citations with similarity scores
- Delete individual documents from knowledge base
- Shows chunk count per document in sidebar

## 🛠️ Tech Stack
- Python 3.10+
- ChromaDB — persistent vector database
- Sentence Transformers (all-MiniLM-L6-v2) — embeddings
- LangChain Text Splitters — recursive chunking
- PyMuPDF (fitz) — PDF text extraction
- Groq API (LLaMA 3.3 70B) — answer generation
- Streamlit — web interface
- python-dotenv — API key management

## 🚀 Setup & Run

### 1. Clone the repo
git clone https://github.com/PrashikSawant/Multi-PDF-RAG-System
cd multi-pdf-rag

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your API key
Create a .env file:
GROQ_API_KEY=your_key_here

### 4. Run
streamlit run app.py

## 🧠 Key Concepts
- Multi-document filtering with ChromaDB $or operator
- Cross-document semantic search
- Compare mode with structured prompting
- Dynamic multiselect UI for document selection
- Grouped context by document for clear citations

## 📁 Project Structure
day13-multi-pdf-rag/
├── app.py              # Streamlit UI, multiselect, modes
├── rag_engine.py       # Multi-doc RAG, $or filtering, compare
├── requirements.txt    # Dependencies
├── .env               # API key (not committed)
├── .gitignore         # Ignores .env, chroma_db, cache
└── chroma_db/         # Auto-created, gitignored

## 🔗 Part of 30-Day AI Engineering Bootcamp
Day 13 of 30 | RAG & Vector Databases Phase
