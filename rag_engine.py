import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import fitz
import os
import hashlib

load_dotenv()

# initialize
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="multi_pdf_rag",
    metadata={"hnsw:space": "cosine"}
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text(uploaded_file):
    filename = uploaded_file.name
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    return ""

def get_doc_id(filename):
    return hashlib.md5(filename.encode()).hexdigest()[:8]

def document_exists(filename):
    results = collection.get(where={"filename": filename})
    return len(results["ids"]) > 0

def add_document(uploaded_file):
    filename = uploaded_file.name
    if document_exists(filename):
        return f"'{filename}' already exists in the database."

    text = extract_text(uploaded_file)
    if not text.strip():
        return "Could not extract text from this file."

    chunks = splitter.split_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    doc_id = get_doc_id(filename)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"filename": filename, "chunk": i} 
                   for i in range(len(chunks))]
    )
    return f"✅ '{filename}' added — {len(chunks)} chunks stored."

def get_documents_list():
    if collection.count() == 0:
        return []
    results = collection.get()
    return list(set(m["filename"] for m in results["metadatas"]))

def delete_document(filename):
    results = collection.get(where={"filename": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return f"✅ '{filename}' deleted."
    return "Document not found."

def get_doc_stats(filename):
    """Get chunk count for a specific document"""
    results = collection.get(where={"filename": filename})
    return len(results["ids"])

def semantic_search(query, selected_docs=None, top_k=5):
    """Search with optional multi-document filter"""
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode([query]).tolist()

    # build where clause for multiple docs
    if selected_docs and len(selected_docs) == 1:
        where = {"filename": selected_docs[0]}
    elif selected_docs and len(selected_docs) > 1:
        where = {"$or": [{"filename": d} for d in selected_docs]}
    else:
        where = None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        where=where
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metadatas, distances))

def answer_question(query, results, mode="qa"):
    """Generate answer — qa mode or compare mode"""
    if not results:
        return "No relevant content found. Upload some documents first."

    # group chunks by document
    doc_chunks = {}
    for doc, meta, _ in results:
        fname = meta["filename"]
        if fname not in doc_chunks:
            doc_chunks[fname] = []
        doc_chunks[fname].append(doc)

    # build labeled context
    context = ""
    for fname, chunks in doc_chunks.items():
        context += f"\n--- {fname} ---\n"
        context += "\n".join(chunks)
        context += "\n"

    if mode == "compare":
        prompt = f"""You are an expert analyst. Compare and contrast 
the following documents based on the question.
Structure your answer clearly by document.
Always reference which document supports each point.

Documents:
{context}

Question: {query}

Comparison:"""
    else:
        prompt = f"""You are a helpful assistant answering questions 
about uploaded documents.
Use ONLY the context below. 
If the answer isn't found, say so clearly.
Always cite which document your answer came from.

Documents:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content