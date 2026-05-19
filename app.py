"""
NoteNest — Day 2
app.py — Streamlit Chat UI

Run with:
    streamlit run app.py

Features in this file:
  - Clean chat interface with NoteNest branding
  - Full conversation history with session state
  - Cited sources (document name + page number) per answer
  - Sidebar with uploaded file info and settings
  - Loading spinner while fetching answers
  - Clear chat button
"""

import streamlit as st
from pathlib import Path
from rag_chain import build_chain


# ─────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoteNest",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Custom CSS — makes NoteNest look polished
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit hamburger and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 4px;
    }

    /* Source citation box */
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4A6CF7;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
        color: #374151;
    }

    /* NoteNest title area */
    .notenest-title {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    .notenest-sub {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load the RAG chain (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain():
    """
    Loads the vectorstore and builds the RAG chain once.
    @st.cache_resource keeps it in memory across reruns —
    so you don't wait for the model to load on every message.
    """
    return build_chain()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 NoteNest")
    st.markdown("*Chat with your lecture notes*")
    st.divider()

    # Show vectorstore info
    vectorstore_path = Path("./vectorstore")
    if vectorstore_path.exists():
        st.success("✅ Knowledge base loaded")
        # Count PDFs in data folder
        data_path = Path("./data")
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))
            if pdfs:
                st.markdown(f"**{len(pdfs)} PDF(s) ingested:**")
                for pdf in pdfs:
                    st.markdown(f"• {pdf.name}")
    else:
        st.error("❌ No knowledge base found")
        st.markdown("Run `python ingest.py` first to process your PDFs.")

    st.divider()

    # Settings
    st.markdown("**Settings**")
    show_sources = st.toggle(
        "Show source citations",
        value=True,
        help="Shows which PDF and page each answer came from"
    )
    num_sources = st.slider(
        "Sources to show per answer",
        min_value=1,
        max_value=4,
        value=2,
        help="How many source chunks to display after each answer"
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:12px;color:#9ca3af;'>NoteNest — Day 2<br>"
        "Built with LangChain + ChromaDB</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────
st.markdown('<div class="notenest-title">📚 NoteNest</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="notenest-sub">Ask anything from your lecture notes — '
    'I\'ll answer with citations from your PDFs.</div>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# Session state — persists across reruns
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────
# Show welcome message if no chat yet
# ─────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm **NoteNest**, your AI study assistant.\n\n"
            "I've read all your uploaded lecture notes. Ask me anything — "
            "definitions, explanations, comparisons, or exam prep questions. "
            "I'll always tell you exactly which PDF and page my answer came from.\n\n"
            "**Try asking:**\n"
            "- *What is a binary search tree?*\n"
            "- *Explain process scheduling in OS*\n"
            "- *What are the ACID properties in DBMS?*"
        )


# ─────────────────────────────────────────────
# Render existing chat history
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Re-render sources for past assistant messages
        if message["role"] == "assistant" and show_sources and "sources" in message:
            if message["sources"]:
                with st.expander(f"📄 {len(message['sources'])} source(s) from your notes"):
                    for i, src in enumerate(message["sources"][:num_sources]):
                        source_file = Path(src["source"]).name
                        page = src["page"]
                        preview = src["content"][:250].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i+1}:</strong> {source_file} — Page {page + 1}<br>'
                            f'<span style="color:#6b7280">"{preview}..."</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )


# ─────────────────────────────────────────────
# Chat input and response
# ─────────────────────────────────────────────
if question := st.chat_input("Ask a question about your notes..."):

    # Check vectorstore exists before trying to answer
    if not Path("./vectorstore").exists():
        st.error("No knowledge base found. Run `python ingest.py` first.")
        st.stop()

    # Add user message to chat
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
                answer = f"Something went wrong: {str(e)}"
                source_docs = []

        # Display the answer
        st.markdown(answer)

        # Build source metadata for storage and display
        sources = []
        for doc in source_docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page":   doc.metadata.get("page", 0),
                "content": doc.page_content,
            })

        # Display source citations
        if show_sources and sources:
            with st.expander(f"📄 {len(sources[:num_sources])} source(s) from your notes"):
                for i, src in enumerate(sources[:num_sources]):
                    source_file = Path(src["source"]).name
                    page = src["page"]
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

    # Update conversation history for the chain's memory
    st.session_state.chat_history.append((question, answer))

    # Keep only the last 5 turns in memory (same as MEMORY_WINDOW)
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]
