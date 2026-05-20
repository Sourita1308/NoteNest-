"""
NoteNest — Day 3
ingest.py — Updated Ingestion Pipeline

New in Day 3:
  - Pinecone cloud vector DB (replaces ChromaDB local)
  - Accepts uploaded files directly (for Streamlit UI upload)
  - Clear + re-ingest functionality
  - Falls back to ChromaDB if no Pinecone key set

Usage:
    python ingest.py                        # ingest from ./data folder
    python ingest.py --clear                # wipe Pinecone index + re-ingest
"""

import os
import argparse
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = "notenest"   # your Pinecone index name


def get_embedding_model() -> HuggingFaceEmbeddings:
    print("🤖  Loading embedding model...")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅  Embedding model ready")
    return model


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️   Split into {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# Pinecone vectorstore
# ─────────────────────────────────────────────
def get_pinecone_vectorstore(embedding_model, create_index=False):
    """
    Connects to Pinecone and returns a LangChain vectorstore.
    Creates the index automatically if it doesn't exist.
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY not found in .env\n"
            "Get a free key at: app.pinecone.io\n"
            "Or remove it to use ChromaDB instead."
        )

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"📦  Creating Pinecone index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,           # matches all-MiniLM-L6-v2 output
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
        print("✅  Pinecone index created")
    else:
        print(f"✅  Pinecone index '{PINECONE_INDEX}' found")

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embedding_model,
        pinecone_api_key=api_key,
    )
    return vectorstore


def clear_pinecone_index():
    """Wipe all vectors from the Pinecone index (for re-ingestion)."""
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return

    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        index = pc.Index(PINECONE_INDEX)
        index.delete(delete_all=True)
        print(f"🗑️   Cleared all vectors from '{PINECONE_INDEX}'")


# ─────────────────────────────────────────────
# ChromaDB fallback
# ─────────────────────────────────────────────
def get_chroma_vectorstore(chunks, embedding_model, persist_dir="./vectorstore"):
    """Falls back to ChromaDB if no Pinecone key is set."""
    from langchain_community.vectorstores import Chroma

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="study_notes",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"✅  Stored in ChromaDB at {persist_dir}/")
    return vectorstore


# ─────────────────────────────────────────────
# Main ingest functions
# ─────────────────────────────────────────────
def ingest_files(uploaded_files: list, clear_first: bool = False) -> dict:
    """
    Ingest a list of uploaded file objects (from Streamlit file_uploader).
    Each file object must have .name and .read() method.

    Returns: {"status": "success", "chunks": N, "files": [...names]}
    """
    print("\n" + "="*50)
    print("  NoteNest — Ingesting uploaded files")
    print("="*50)

    if not uploaded_files:
        return {"status": "error", "message": "No files provided"}

    # Save uploaded files to temp directory and load them
    documents = []
    file_names = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            # Write to temp file
            tmp_path = Path(tmp_dir) / uploaded_file.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF
            loader = PyPDFLoader(str(tmp_path))
            docs = loader.load()

            # Fix metadata source to show original filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)
            file_names.append(uploaded_file.name)
            print(f"  ✅  {uploaded_file.name} — {len(docs)} pages")

    print(f"\n📄  Total pages loaded: {len(documents)}")

    # Split
    chunks = split_documents(documents)

    # Embed
    embedding_model = get_embedding_model()

    # Store — Pinecone if key exists, else ChromaDB
    use_pinecone = bool(os.getenv("PINECONE_API_KEY"))

    if use_pinecone:
        if clear_first:
            clear_pinecone_index()
        vectorstore = get_pinecone_vectorstore(embedding_model)
        vectorstore.add_documents(chunks)
        print(f"✅  Uploaded {len(chunks)} chunks to Pinecone")
    else:
        if clear_first:
            import shutil
            if Path("./vectorstore").exists():
                shutil.rmtree("./vectorstore")
                print("🗑️   Cleared ChromaDB vectorstore")
        vectorstore = get_chroma_vectorstore(chunks, embedding_model)

    return {
        "status": "success",
        "chunks": len(chunks),
        "files": file_names,
        "backend": "Pinecone" if use_pinecone else "ChromaDB",
    }


def ingest_from_folder(data_dir: str = "./data", clear_first: bool = False):
    """Ingest all PDFs from a local folder (original Day 1 behaviour)."""
    print("\n" + "="*50)
    print("  NoteNest — Ingesting from folder")
    print("="*50)

    data_path = Path(data_dir)
    if not data_path.exists() or not list(data_path.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}'.\n"
            "Add PDF files and retry."
        )

    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    print(f"📂  Loaded {len(documents)} pages from {data_dir}")

    chunks = split_documents(documents)
    embedding_model = get_embedding_model()

    use_pinecone = bool(os.getenv("PINECONE_API_KEY"))

    if use_pinecone:
        if clear_first:
            clear_pinecone_index()
        vectorstore = get_pinecone_vectorstore(embedding_model)
        vectorstore.add_documents(chunks)
        print(f"✅  Uploaded {len(chunks)} chunks to Pinecone")
    else:
        if clear_first:
            import shutil
            if Path("./vectorstore").exists():
                shutil.rmtree("./vectorstore")
        vectorstore = get_chroma_vectorstore(chunks, embedding_model)

    print("\n✅  Ingestion complete!")
    print("   Run: streamlit run app.py")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoteNest ingestion pipeline")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing index before ingesting")
    args = parser.parse_args()
    ingest_from_folder(data_dir=args.data_dir, clear_first=args.clear)
