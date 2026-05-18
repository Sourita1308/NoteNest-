"""
RAG Study Chatbot — Day 1
Ingestion Pipeline: PDF → Chunks → Embeddings → ChromaDB

Run this once per new set of PDFs:
    python ingest.py
Or with a custom data folder:
    python ingest.py --data_dir ./my_notes --persist_dir ./vectorstore
"""

import os
import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ─────────────────────────────────────────────
# Configuration — tweak these as needed
# ─────────────────────────────────────────────
CHUNK_SIZE      = 512    # tokens per chunk (512 is ideal for RAG)
CHUNK_OVERLAP   = 64     # overlap keeps context across chunk boundaries
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, fast, offline
TOP_K_DEFAULT   = 4      # how many chunks to retrieve at query time


def load_pdfs(data_dir: str) -> list:
    """
    Load all PDFs from a directory.
    Returns a list of LangChain Document objects (one per page).
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found.\n"
            f"Create it and drop your PDF notes inside:\n"
            f"  mkdir {data_dir}"
        )

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{data_dir}'.\n"
            f"Add your lecture notes / textbook PDFs there and re-run."
        )

    print(f"\n📂  Found {len(pdf_files)} PDF(s) in '{data_dir}':")
    for f in pdf_files:
        print(f"     • {f.name}")

    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()

    print(f"\n✅  Loaded {len(documents)} pages total")
    return documents


def split_documents(documents: list) -> list:
    """
    Split pages into overlapping chunks.

    Why RecursiveCharacterTextSplitter?
      It tries to split on paragraph breaks first (\n\n),
      then line breaks (\n), then sentences (.), then words.
      This keeps semantically related text together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Log chunk stats so you can tune CHUNK_SIZE if needed
    sizes = [len(c.page_content) for c in chunks]
    avg   = sum(sizes) / len(sizes) if sizes else 0
    print(f"\n✂️   Split into {len(chunks)} chunks")
    print(f"     Avg chunk size : {avg:.0f} chars")
    print(f"     Min / Max      : {min(sizes)} / {max(sizes)} chars")

    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace embedding model (downloads on first run, ~80 MB).

    all-MiniLM-L6-v2 facts:
      • 384-dimensional vectors
      • Extremely fast (CPU-friendly)
      • Excellent semantic similarity for English text
      • 100% free — no API key needed
    """
    print(f"\n🤖  Loading embedding model: {EMBEDDING_MODEL}")
    print("     (First run downloads ~80 MB — subsequent runs use cache)")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity friendly
    )

    print("✅  Embedding model ready")
    return embedding_model


def store_in_chromadb(chunks: list, embedding_model, persist_dir: str) -> Chroma:
    """
    Embed all chunks and store them in ChromaDB.

    ChromaDB persists to disk at persist_dir so you don't re-ingest on every run.
    The collection is named 'study_notes' — you can change this.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📦  Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"     This may take 1–3 minutes for large document sets.\n")

    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="study_notes",
        collection_metadata={"hnsw:space": "cosine"},  # cosine distance for retrieval
    )

    elapsed = time.time() - start
    print(f"\n✅  Stored {len(chunks)} chunks in ChromaDB")
    print(f"     Time taken     : {elapsed:.1f}s")
    print(f"     Persisted at   : {persist_dir}/")
    print(f"     Collection     : study_notes")

    return vectorstore


def verify_vectorstore(vectorstore: Chroma) -> None:
    """
    Quick sanity check — run a test similarity search to confirm everything works.
    """
    print("\n🔍  Running verification search...")

    test_query = "What is the main topic of these notes?"
    results = vectorstore.similarity_search(test_query, k=2)

    if results:
        print(f"✅  Retrieval working! Sample chunk from your notes:\n")
        sample = results[0].page_content[:200].replace("\n", " ")
        source = results[0].metadata.get("source", "unknown")
        page   = results[0].metadata.get("page", "?")
        print(f'     "{sample}..."')
        print(f"     Source: {Path(source).name}, page {page}")
    else:
        print("⚠️  No results returned — check your PDFs contain readable text")


def ingest(data_dir: str = "./data", persist_dir: str = "./vectorstore") -> None:
    """
    Full ingestion pipeline:
      1. Load PDFs  →  2. Split  →  3. Embed  →  4. Store  →  5. Verify
    """
    print("=" * 55)
    print("  RAG Study Chatbot — Ingestion Pipeline")
    print("=" * 55)

    # Step 1: Load
    documents = load_pdfs(data_dir)

    # Step 2: Split
    chunks = split_documents(documents)

    # Step 3: Embed
    embedding_model = create_embeddings()

    # Step 4: Store
    vectorstore = store_in_chromadb(chunks, embedding_model, persist_dir)

    # Step 5: Verify
    verify_vectorstore(vectorstore)

    print("\n" + "=" * 55)
    print("  ✅  Ingestion complete! Next step:")
    print("  Run:  streamlit run app.py")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB for the RAG Study Chatbot"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Folder containing your PDF notes (default: ./data)"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./vectorstore",
        help="Where to save the ChromaDB vector store (default: ./vectorstore)"
    )
    args = parser.parse_args()

    ingest(data_dir=args.data_dir, persist_dir=args.persist_dir)
