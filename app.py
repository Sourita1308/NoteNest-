"""
NoteNest — Day 4
app.py — Full UI with Voice Input

New in Day 4:
  - 🎙 Mic button next to chat input
  - Record audio → Whisper transcribes → auto-fills question
  - Works with local Whisper (free) or Whisper API (paid)
  - Live recording timer
  - Graceful fallback if mic packages not installed
  - All Day 3 features retained

Run with:
    streamlit run app.py
"""

import os
import time
import streamlit as st
from pathlib import Path
from rag_chain import build_chain
from ingest import ingest_files
from voice_input import check_dependencies, transcribe_audio_file


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
# CSS
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
        margin-bottom: 6px;
    }

    .pill-green  { background:#d1fae5; color:#065f46; }
    .pill-blue   { background:#dbeafe; color:#1e40af; }
    .pill-red    { background:#fee2e2; color:#991b1b; }
    .pill-orange { background:#ffedd5; color:#9a3412; }
    .pill-purple { background:#ede9fe; color:#5b21b6; }

    .upload-hint {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .voice-hint {
        font-size: 12px;
        color: #6b7280;
        padding: 6px 10px;
        background: #f9fafb;
        border-radius: 6px;
        border: 1px dashed #d1d5db;
        margin-top: 6px;
    }

    .recording-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 13px;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "chat_history": [],
    "ingested_files": [],
    "kb_ready": Path("./vectorstore").exists(),
    "voice_transcript": "",
    "is_recording": False,
    "recorder": None,
    "record_start_time": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# Check voice dependencies once
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_voice_deps():
    return check_dependencies()

@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()

voice_deps = get_voice_deps()
voice_available = voice_deps["sounddevice"] and (
    voice_deps["whisper_local"] or voice_deps["whisper_api"]
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 NoteNest")
    st.markdown("*Your AI study assistant*")
    st.divider()

    # ── Knowledge base status ──
    st.markdown("**Knowledge Base**")
    using_pinecone = bool(os.getenv("PINECONE_API_KEY"))
    backend_label = "Pinecone ☁️" if using_pinecone else "ChromaDB 💾"
    backend_class = "pill-blue" if using_pinecone else "pill-green"

    st.markdown(
        f'<span class="status-pill {backend_class}">Backend: {backend_label}</span>',
        unsafe_allow_html=True
    )

    if st.session_state.kb_ready:
        st.markdown('<span class="status-pill pill-green">✅ Knowledge base ready</span>',
                    unsafe_allow_html=True)
        if st.session_state.ingested_files:
            st.markdown(f"**{len(st.session_state.ingested_files)} file(s) loaded:**")
            for f in st.session_state.ingested_files:
                st.markdown(f"• {f}")
    else:
        st.markdown('<span class="status-pill pill-red">❌ No knowledge base</span>',
                    unsafe_allow_html=True)

    st.divider()

    # ── PDF Upload ──
    st.markdown("**Upload Your Notes**")
    st.markdown('<div class="upload-hint">Drop lecture PDFs here</div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        ingest_btn = st.button("📥 Ingest", use_container_width=True,
                               disabled=not uploaded_files)
    with col2:
        clear_btn = st.button("🗑️ Clear KB", use_container_width=True)

    if ingest_btn and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
            progress = st.progress(0, text="Starting...")
            try:
                progress.progress(20, text="Loading PDFs...")
                result = ingest_files(uploaded_files, clear_first=False)
                progress.progress(100, text="Done!")
                if result["status"] == "success":
                    st.session_state.ingested_files.extend(result["files"])
                    st.session_state.kb_ready = True
                    st.cache_resource.clear()
                    st.success(f"✅ {result['chunks']} chunks ingested into {result['backend']}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

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
            st.success("🗑️ Cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.divider()

    # ── Voice settings ──
    st.markdown("**Voice Input 🎙️**")

    if voice_available:
        st.markdown('<span class="status-pill pill-purple">✅ Voice ready</span>',
                    unsafe_allow_html=True)

        use_api_whisper = st.toggle(
            "Use Whisper API (faster)",
            value=False,
            help="ON = Whisper API (needs OPENAI_API_KEY, costs $0.006/min)\nOFF = Local Whisper (free, offline)"
        )

        whisper_model = st.selectbox(
            "Local Whisper model",
            ["tiny", "base", "small"],
            index=1,
            help="base = best balance of speed and accuracy",
            disabled=use_api_whisper,
        )
        st.markdown(
            '<div class="upload-hint">Click 🎙 in the chat area to record</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<span class="status-pill pill-orange">⚠️ Not installed</span>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="voice-hint">Run in terminal:<br>'
            '<code>pip install sounddevice openai-whisper</code><br>'
            'Then restart the app.</div>',
            unsafe_allow_html=True
        )
        use_api_whisper = False
        whisper_model = "base"

    st.divider()

    # ── Chat settings ──
    st.markdown("**Chat Settings**")
    show_sources = st.toggle("Show citations", value=True)
    num_sources = st.slider("Sources per answer", 1, 4, 2)

    if st.button("💬 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.voice_transcript = ""
        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:11px;color:#9ca3af;'>NoteNest Day 4<br>"
        f"Backend: {backend_label}<br>"
        f"Voice: {'Whisper API' if use_api_whisper else f'Local ({whisper_model})'}</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("## 📚 NoteNest")
st.markdown("*Ask anything from your lecture notes — by typing or speaking*")

# ── Welcome ──
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.kb_ready:
            voice_tip = " or click **🎙 Record** below to ask by voice" if voice_available else ""
            st.markdown(
                f"👋 Hi! I'm **NoteNest**.\n\n"
                f"Your notes are loaded. Type a question{voice_tip}.\n\n"
                f"**Try asking:**\n"
                f"- *Explain CPU scheduling algorithms*\n"
                f"- *What is a deadlock?*\n"
                f"- *Summarise key points from my notes*"
            )
        else:
            st.markdown(
                "👋 Hi! I'm **NoteNest**.\n\n"
                "📂 Upload your lecture PDFs in the sidebar → click **Ingest**.\n\n"
                "Then ask me anything!"
            )

# ── Chat history ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (message["role"] == "assistant" and show_sources
                and "sources" in message and message["sources"]):
            with st.expander(f"📄 {min(len(message['sources']), num_sources)} source(s)"):
                for i, src in enumerate(message["sources"][:num_sources]):
                    source_file = Path(src["source"]).name
                    page = src.get("page", 0)
                    preview = src["content"][:250].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><strong>Source {i+1}:</strong> '
                        f'{source_file} — Page {page+1}<br>'
                        f'<span style="color:#6b7280">"{preview}..."</span></div>',
                        unsafe_allow_html=True
                    )


# ─────────────────────────────────────────────
# VOICE INPUT SECTION
# ─────────────────────────────────────────────
if voice_available and st.session_state.kb_ready:
    st.markdown("---")

    # ── Streamlit native audio recorder (simplest approach) ──
    st.markdown("**🎙️ Voice Input** — Record your question")

    col_rec, col_status = st.columns([1, 3])

    with col_rec:
        audio_data = st.audio_input(
            "Click to record",
            label_visibility="collapsed",
            key="audio_recorder"
        )

    with col_status:
        if audio_data is not None:
            st.markdown(
                '<div class="recording-badge">🎧 Processing voice...</div>',
                unsafe_allow_html=True
            )

    # Process recorded audio
    if audio_data is not None:
        with st.spinner("Transcribing with Whisper..."):
            try:
                audio_bytes = audio_data.read()
                transcript = transcribe_audio_file(
                    audio_bytes,
                    use_api=use_api_whisper
                )
                if transcript:
                    st.session_state.voice_transcript = transcript
                    st.success(f"🎤 Heard: *\"{transcript}\"*")
                else:
                    st.warning("Couldn't hear anything clearly. Try again.")
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

# Pre-fill with voice transcript if available
prefill_value = st.session_state.get("voice_transcript", "")

question = st.chat_input(
    "Ask a question... (or use 🎙 above to speak)",
    disabled=not st.session_state.kb_ready,
)

# If no typed question but we have a voice transcript, use that
if not question and prefill_value:
    question = prefill_value
    st.session_state.voice_transcript = ""   # clear after use

# ── Process question ──
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

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

        st.markdown(answer)

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
                        f'<div class="source-box"><strong>Source {i+1}:</strong> '
                        f'{source_file} — Page {page+1}<br>'
                        f'<span style="color:#6b7280">"{preview}..."</span></div>',
                        unsafe_allow_html=True
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.append((question, answer))
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]

    # Clear voice transcript
    st.session_state.voice_transcript = ""
