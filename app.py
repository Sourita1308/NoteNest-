"""
NoteNest — Day 3
app.py — Full UI with Multi-File Upload

New in Day 3:
  - Upload PDFs directly in the sidebar (no more data/ folder needed)
  - Ingest button processes uploads on the spot
  - Clear knowledge base button
  - Shows Pinecone or ChromaDB backend status
  - Progress bar during ingestion
  - All Day 2 features retained

Run with:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from rag_chain import build_chain
from ingest import ingest_files


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoteNest",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4A6CF7;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
        color: #374151;
    }

    .status-pill {
        display: inline-block;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .pill-green {
        background: #d1fae5;
        color: #065f46;
    }

    .pill-blue {
        background: #dbeafe;
        color: #1e40af;
    }

    .pill-red {
        background: #fee2e2;
        color: #991b1b;
    }

    .upload-hint {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "kb_ready" not in st.session_state:
    # Check if a vectorstore already exists from Day 1/2
    st.session_state.kb_ready = (
        Path("./vectorstore").exists() or
        bool(st.session_state.ingested_files)
    )


# ─────────────────────────────────────────────
# Load chain (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 NoteNest")
    st.markdown("*Your AI study assistant*")
    st.divider()

    # ── Knowledge base status ──
    st.markdown("**Knowledge Base**")

    import os
    using_pinecone = bool(os.getenv("PINECONE_API_KEY"))
    backend_label = "Pinecone ☁️" if using_pinecone else "ChromaDB 💾"
    backend_class = "pill-blue" if using_pinecone else "pill-green"

    st.markdown(
        f'<span class="status-pill {backend_class}">Backend: {backend_label}</span>',
        unsafe_allow_html=True
    )

    if st.session_state.kb_ready:
        st.markdown(
            '<span class="status-pill pill-green">✅ Knowledge base ready</span>',
            unsafe_allow_html=True
        )
        if st.session_state.ingested_files:
            st.markdown(f"**{len(st.session_state.ingested_files)} file(s) loaded:**")
            for f in st.session_state.ingested_files:
                st.markdown(f"• {f}")
    else:
        st.markdown(
            '<span class="status-pill pill-red">❌ No knowledge base yet</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="upload-hint">Upload PDFs below to get started</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── PDF Upload section ──
    st.markdown("**Upload Your Notes**")
    st.markdown(
        '<div class="upload-hint">Drop lecture PDFs here — NoteNest will read them instantly</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Select one or more PDF files from your computer"
    )

    col1, col2 = st.columns(2)

    with col1:
        ingest_btn = st.button(
            "📥 Ingest",
            use_container_width=True,
            help="Process uploaded PDFs and add to knowledge base",
            disabled=not uploaded_files
        )

    with col2:
        clear_btn = st.button(
            "🗑️ Clear KB",
            use_container_width=True,
            help="Wipe the knowledge base and start fresh"
        )

    # ── Handle Ingest ──
    if ingest_btn and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
            progress = st.progress(0, text="Starting ingestion...")

            try:
                progress.progress(20, text="Loading PDFs...")
                result = ingest_files(uploaded_files, clear_first=False)
                progress.progress(80, text="Storing embeddings...")

                if result["status"] == "success":
                    progress.progress(100, text="Done!")

                    # Update session state
                    new_files = result["files"]
                    st.session_state.ingested_files.extend(new_files)
                    st.session_state.kb_ready = True

                    # Clear cached chain so it reloads with new data
                    st.cache_resource.clear()

                    st.success(
                        f"✅ Ingested {result['chunks']} chunks from "
                        f"{len(new_files)} file(s) into {result['backend']}"
                    )
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {result.get('message')}")

            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
                progress.empty()

    # ── Handle Clear ──
    if clear_btn:
        try:
            if using_pinecone:
                from ingest import clear_pinecone_index
                clear_pinecone_index()
            else:
                import shutil
                if Path("./vectorstore").exists():
                    shutil.rmtree("./vectorstore")

            st.session_state.ingested_files = []
            st.session_state.kb_ready = False
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.cache_resource.clear()
            st.success("🗑️ Knowledge base cleared")
            st.rerun()

        except Exception as e:
            st.error(f"Error clearing: {str(e)}")

    st.divider()

    # ── Chat settings ──
    st.markdown("**Chat Settings**")
    show_sources = st.toggle("Show source citations", value=True)
    num_sources = st.slider("Sources per answer", 1, 4, 2)

    st.divider()

    # ── Clear chat ──
    if st.button("💬 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#9ca3af;'>"
        "NoteNest — Day 3<br>"
        f"Backend: {backend_label}<br>"
        "LangChain · HuggingFace · Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────
st.markdown("## 📚 NoteNest")
st.markdown("*Ask anything from your lecture notes — answers with citations*")

# ── Welcome message ──
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.kb_ready:
            st.markdown(
                "👋 Hi! I'm **NoteNest**.\n\n"
                "Your notes are loaded and ready. Ask me anything — "
                "I'll answer directly from your PDFs and show you exactly "
                "which page the answer came from.\n\n"
                "**Try asking:**\n"
                "- *Explain CPU scheduling algorithms*\n"
                "- *What is the difference between process and thread?*\n"
                "- *Summarise the key points from my notes*"
            )
        else:
            st.markdown(
                "👋 Hi! I'm **NoteNest**.\n\n"
                "📂 **To get started:** upload your lecture PDFs in the "
                "sidebar on the left, then click **Ingest**.\n\n"
                "Once ingested, ask me anything from your notes!"
            )

# ── Render chat history ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (message["role"] == "assistant"
                and show_sources
                and "sources" in message
                and message["sources"]):
            with st.expander(f"📄 {min(len(message['sources']), num_sources)} source(s)"):
                for i, src in enumerate(message["sources"][:num_sources]):
                    source_file = Path(src["source"]).name
                    page = src.get("page", 0)
                    preview = src["content"][:250].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Source {i+1}:</strong> {source_file} — Page {page + 1}<br>'
                        f'<span style="color:#6b7280">"{preview}..."</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ── Chat input ──
if question := st.chat_input(
    "Ask a question about your notes...",
    disabled=not st.session_state.kb_ready
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                chain = get_chain()
                result = chain({
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                })
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

            except Exception as e:
                answer = (
                    f"Something went wrong: {str(e)}\n\n"
                    "Try re-ingesting your PDFs or restarting the app."
                )
                source_docs = []

        st.markdown(answer)

        # Build + show sources
        sources = [
            {
                "source":  doc.metadata.get("source", "Unknown"),
                "page":    doc.metadata.get("page", 0),
                "content": doc.page_content,
            }
            for doc in source_docs
        ]

        if show_sources and sources:
            with st.expander(f"📄 {min(len(sources), num_sources)} source(s)"):
                for i, src in enumerate(sources[:num_sources]):
                    source_file = Path(src["source"]).name
                    page = src.get("page", 0)
                    preview = src["content"][:250].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Source {i+1}:</strong> {source_file} — Page {page + 1}<br>'
                        f'<span style="color:#6b7280">"{preview}..."</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    st.session_state.chat_history.append((question, answer))
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]
