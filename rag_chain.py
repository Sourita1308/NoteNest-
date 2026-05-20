"""
NoteNest — Day 3
rag_chain.py — Updated Query Pipeline

New in Day 3:
  - Auto-detects Pinecone vs ChromaDB based on .env
  - Same interface as Day 2 — app.py needs no changes
"""

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

PERSIST_DIR     = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = "notenest"
TOP_K_CHUNKS    = 4
TEMPERATURE     = 0.2

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


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore():
    """
    Auto-detects which vectorstore to load:
    - Pinecone if PINECONE_API_KEY is in .env
    - ChromaDB otherwise
    """
    embedding_model = get_embedding_model()
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if pinecone_key:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore

        print("📚  Connecting to Pinecone knowledge base...")
        pc = Pinecone(api_key=pinecone_key)
        existing = [i.name for i in pc.list_indexes()]

        if PINECONE_INDEX not in existing:
            raise RuntimeError(
                f"Pinecone index '{PINECONE_INDEX}' not found.\n"
                "Run: python ingest.py  to create and populate it."
            )

        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX,
            embedding=embedding_model,
            pinecone_api_key=pinecone_key,
        )
        index = pc.Index(PINECONE_INDEX)
        stats = index.describe_index_stats()
        count = stats.get("total_vector_count", "unknown")
        print(f"✅  Pinecone loaded — {count} chunks indexed")

    else:
        from langchain_community.vectorstores import Chroma

        if not Path(PERSIST_DIR).exists():
            raise FileNotFoundError(
                "No vectorstore found. Run: python ingest.py"
            )
        print("📚  Loading ChromaDB knowledge base...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name="study_notes",
        )
        count = vectorstore._collection.count()
        print(f"✅  ChromaDB loaded — {count} chunks")

    return vectorstore


def get_llm():
    gemini_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("🤖  LLM: Gemini 2.5 Flash (free)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=TEMPERATURE,
            convert_system_message_to_human=True,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        print("🤖  LLM: GPT-3.5-turbo")
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_key,
            temperature=TEMPERATURE,
        )
    else:
        raise EnvironmentError(
            "No API key found.\n"
            "Add GOOGLE_API_KEY or OPENAI_API_KEY to your .env file."
        )


def build_chain() -> ConversationalRetrievalChain:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS},
    )
    llm = get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": NOTENEST_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    print("✅  NoteNest query chain ready\n")
    return chain
