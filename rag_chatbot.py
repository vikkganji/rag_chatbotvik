import os
import glob
import uuid
import textwrap
from typing import List, Tuple
from tqdm import tqdm

# --- Storage & Embeddings ---
import chromadb
from chromadb.utils import embedding_functions

# --- For PDF loading ---
from pypdf import PdfReader

# --- For API calls ---
import requests

# =========================
# Config
# =========================
DATA_DIR = "data"
COLLECTION_NAME = "knowledge_base"

# ✅ Your Groq API key (use env var in production)
GROQ_API_KEY = ""

# =========================
# Chunking Helper
# =========================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# =========================
# Document Loader
# =========================
def load_pdfs_from_dir(data_dir: str) -> List[Tuple[str, str]]:
    """Load PDFs and return list of (filename, text)."""
    docs = []
    for pdf_path in glob.glob(os.path.join(data_dir, "*.pdf")):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        docs.append((os.path.basename(pdf_path), text))
    return docs

# =========================
# Ingestion into ChromaDB
# =========================
def ingest_documents(docs: List[Tuple[str, str]], collection):
    """Chunk documents and add to ChromaDB."""
    for filename, text in tqdm(docs, desc="Indexing documents"):
        chunks = chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)

# =========================
# Query ChromaDB
# =========================
def query_collection(collection, query: str, n_results: int = 3):
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append(f"[Retrieved: {meta['source']}] {doc[:200]}...")
    return "\n".join(retrieved)

# =========================
# Groq API Caller
# =========================
def call_groq_chat(system_msg: str, user_msg: str) -> str:
    """Call Groq API for chat completions."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        # ✅ Updated model (general-purpose, supported)
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    # If request fails, show Groq’s error message
    if r.status_code != 200:
        raise Exception(f"{r.status_code} {r.reason} → {r.text}")

    res = r.json()
    return res["choices"][0]["message"]["content"].strip()

# =========================
# RAG Pipeline
# =========================
def rag_query(user_query: str, collection) -> str:
    retrieved = query_collection(collection, user_query, n_results=3)
    system_msg = "You are a helpful AI assistant. Use the retrieved documents to answer."
    full_prompt = f"User question: {user_query}\n\nContext:\n{retrieved}"
    try:
        answer = call_groq_chat(system_msg, full_prompt)
        return f"Assistant:\n{answer}\n\n{retrieved}"
    except Exception as e:
        return f"⚠️ Error calling Groq API: {e}\n\n{retrieved}"

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Setup Chroma
    client = chromadb.Client()
    embedder = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    # Load and ingest documents
    docs = load_pdfs_from_dir(DATA_DIR)
    ingest_documents(docs, collection)

    print("RAG system ready. Ask me something!\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print(rag_query(q, collection))
