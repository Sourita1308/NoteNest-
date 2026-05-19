"""
NoteNest — Day 2
rag_chain.py — Query Pipeline

This file builds the brain of NoteNest.
It loads the vectorstore built on Day 1 and creates a
ConversationalRetrievalChain that:
  1. Embeds the user's question
  2. Finds the 4 most relevant chunks from your notes
  3. Sends those chunks + the question to the LLM
  4. Returns a grounded answer with source citations

Usage (called from app.py):
    from rag_chain import build_chain
    chain = build_chain()
    result = chain({"question": "What is a binary tree?", "chat_history": []})
    print(result["answer"])
    print(result["source_documents"])
"""

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PERSIST_DIR     = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CHUNKS    = 4      # how many note chunks to retrieve per question
MEMORY_WINDOW   = 5      # how many past conversation turns to remember
TEMPERATURE     = 0.2    # low = factual, high = creative


# ─────────────────────────────────────────────
# Custom system prompt — the personality of NoteNest
# ─────────────────────────────────────────────
NOTENEST_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are NoteNest, an intelligent study assistant.
Your job is to help students understand their own lecture notes.

STRICT RULES:
- Answer ONLY using the context provided below. Never use outside knowledge.
- If the answer is not in the context, say exactly:
  "I couldn't find that in your notes. Try uploading more PDFs on this topic."
- Always mention which document and page your answer came from.
- Keep answers clear, structured, and student-friendly.
- If a concept is complex, break it into numbered steps.
- Never make up facts, formulas, or definitions.

Context from your notes:
{context}

Conversation so far:
{chat_history}

Student's question: {question}

NoteNest answer:"""
)


def load_vectorstore() -> Chroma:
    """
    Load the ChromaDB vectorstore saved by ingest.py on Day 1.
    Uses the same embedding model so vectors are comparable.
    """
    persist_path = Path(PERSIST_DIR)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at '{PERSIST_DIR}'.\n"
            "Run Day 1 first:  python ingest.py"
        )

    print("📚  Loading NoteNest knowledge base...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name="study_notes",
    )

    doc_count = vectorstore._collection.count()
    print(f"✅  Knowledge base loaded — {doc_count} chunks from your notes")
    return vectorstore


def get_llm():
    """
    Returns the LLM — auto-detects whether to use Gemini or OpenAI
    based on which key is set in your .env file.
    Gemini is free tier; OpenAI is paid.
    """
    gemini_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        # FREE option — Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("🤖  Using LLM: Gemini (free tier)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=TEMPERATURE,
            convert_system_message_to_human=True,
        )

    elif openai_key:
        # Paid option — OpenAI GPT
        from langchain_openai import ChatOpenAI
        print("🤖  Using LLM: OpenAI GPT-3.5-turbo")
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_key,
            temperature=TEMPERATURE,
        )

    else:
        raise EnvironmentError(
            "No API key found in .env file.\n"
            "Add either GOOGLE_API_KEY or OPENAI_API_KEY to your .env file.\n"
            "Free Gemini key: aistudio.google.com/app/apikey"
        )


def build_chain() -> ConversationalRetrievalChain:
    """
    Builds the full RAG query chain:
      vectorstore retriever + LLM + memory + custom prompt

    Returns a chain you call like:
        result = chain({"question": "...", "chat_history": []})
    """
    # Step 1: Load the knowledge base
    vectorstore = load_vectorstore()

    # Step 2: Create a retriever — finds top-K most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",       # cosine similarity search
        search_kwargs={"k": TOP_K_CHUNKS},
    )

    # Step 3: Load the LLM (auto-detects Gemini or OpenAI)
    llm = get_llm()

    # Step 4: Build the chain with our custom NoteNest prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": NOTENEST_PROMPT},
        return_source_documents=True,   # needed for citations
        verbose=False,
    )

    print("✅  NoteNest query chain ready\n")
    return chain
