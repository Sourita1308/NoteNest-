"""
NoteNest — Day 5
app.py — Full Voice Conversation Loop

New in Day 5:
  - NoteNest speaks answers back using gTTS or ElevenLabs
  - Voice toggle in sidebar (enable/disable voice responses)
  - Language selector for TTS output
  - ElevenLabs voice picker when API key is set
  - Auto-play audio after each answer
  - Full voice loop: speak question → get answer → hear answer
  - All Day 4 features retained

Run with:
    streamlit run app.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import base64
import streamlit as st
from pathlib import Path
from rag_chain import build_chain
from ingest import ingest_files
from voice_input import check_dependencies, transcribe_audio_file
from voice_output import speak, get_tts_status, GTTS_LANG_MAP, ELEVENLABS_VOICES


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
    .pill-pink   { background:#fce7f3; color:#9d174d; }

    .upload-hint {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .voice-card {
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 8px 0;
        font-size: 13px;
    }

    .audio-label {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain():
    return build_chain()

@st.cache_resource(show_spinner=False)
def get_voice_deps():
    return check_dependencies()

@st.cache_resource(show_spinner=False)
def get_tts():
    return get_tts_status()


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "chat_history": [],
    "ingested_files": [],
    "kb_ready": Path("./vectorstore").exists(),
    "voice_transcript": "",
    "last_audio_b64": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

voice_deps = get_voice_deps()
tts_status = get_tts()
voice_input_ok  = voice_deps["sounddevice"] and (
    voice_deps["whisper_local"] or voice_deps["whisper_api"]
)
voice_output_ok = tts_status["gtts"] or tts_status["elevenlabs"]


# ─────────────────────────────────────────────
# Helper — auto-play audio in browser
# ─────────────────────────────────────────────
def autoplay_audio(b64_audio: str):
    """Inject an auto-playing hidden audio element into the page."""
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 NoteNest")
    st.markdown("*Your AI study assistant*")
    st.divider()

    # ── Knowledge base ──
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
            for f in st.session_state.ingested_files:
                st.markdown(f"• {f}")
    else:
        st.markdown('<span class="status-pill pill-red">❌ No knowledge base</span>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Upload ──
    st.markdown("**Upload Your Notes**")
    st.markdown('<div class="upload-hint">Drop lecture PDFs here</div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"],
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
            prog = st.progress(0, "Starting...")
            try:
                prog.progress(20, "Loading PDFs...")
                result = ingest_files(uploaded_files, clear_first=False)
                prog.progress(100, "Done!")
                if result["status"] == "success":
                    st.session_state.ingested_files.extend(result["files"])
                    st.session_state.kb_ready = True
                    st.cache_resource.clear()
                    st.success(f"✅ {result['chunks']} chunks → {result['backend']}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if clear_btn:
        try:
            if using_pinecone:
                from ingest import clear_pinecone_index
                clear_pinecone_index()
            else:
                import shutil
                if Path("./vectorstore").exists():
                    shutil.rmtree("./vectorstore")
            for k in ["ingested_files","kb_ready","messages","chat_history"]:
                st.session_state[k] = [] if isinstance(
                    st.session_state[k], list) else False
            st.cache_resource.clear()
            st.success("🗑️ Cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # ── Voice Input settings ──
    st.markdown("**Voice Input 🎙️**")
    if voice_input_ok:
        st.markdown('<span class="status-pill pill-purple">✅ Whisper ready</span>',
                    unsafe_allow_html=True)
        use_api_whisper = st.toggle("Use Whisper API", value=False,
            help="ON = paid API | OFF = free local")
        whisper_model = st.selectbox("Whisper model",
            ["tiny","base","small"], index=1,
            disabled=use_api_whisper)
    else:
        st.markdown('<span class="status-pill pill-orange">⚠️ Not installed</span>',
                    unsafe_allow_html=True)
        st.code("pip install sounddevice openai-whisper", language="bash")
        use_api_whisper = False
        whisper_model = "base"

    st.divider()

    # ── Voice Output settings ──
    st.markdown("**Voice Output 🔊**")

    if voice_output_ok:
        engine = tts_status["active_engine"]
        engine_class = "pill-pink" if engine == "ElevenLabs" else "pill-green"
        st.markdown(
            f'<span class="status-pill {engine_class}">✅ {engine}</span>',
            unsafe_allow_html=True
        )

        voice_responses = st.toggle(
            "Speak answers aloud",
            value=True,
            help="NoteNest reads answers back to you"
        )

        # Language selector (gTTS)
        if engine != "ElevenLabs":
            tts_lang_name = st.selectbox(
                "Response language",
                list(GTTS_LANG_MAP.keys()),
                index=0,
                help="Language for spoken responses"
            )
            tts_lang_code = GTTS_LANG_MAP[tts_lang_name]
            selected_voice = None
        else:
            # ElevenLabs voice picker
            selected_voice = st.selectbox(
                "Voice",
                list(ELEVENLABS_VOICES.keys()),
                index=0,
            )
            tts_lang_code = "en"

        # Volume / speed note
        st.markdown(
            '<div class="upload-hint">Answers auto-play after generation</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown('<span class="status-pill pill-orange">⚠️ Not installed</span>',
                    unsafe_allow_html=True)
        st.code("pip install gTTS", language="bash")
        voice_responses = False
        tts_lang_code = "en"
        selected_voice = None

    st.divider()

    # ── Chat settings ──
    st.markdown("**Chat Settings**")
    show_sources = st.toggle("Show citations", value=True)
    num_sources  = st.slider("Sources per answer", 1, 4, 2)

    if st.button("💬 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.voice_transcript = ""
        st.session_state.last_audio_b64 = None
        st.rerun()

    st.divider()
    st.markdown(
        f"<div style='font-size:11px;color:#9ca3af;'>NoteNest Day 5<br>"
        f"Backend: {backend_label}<br>"
        f"TTS: {tts_status['active_engine']}</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("## 📚 NoteNest")
st.markdown("*Ask by typing or speaking — NoteNest answers and reads back to you*")

# ── Welcome ──
if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.kb_ready:
            st.markdown(
                "👋 Hi! I'm **NoteNest**.\n\n"
                "Your notes are loaded. I can now **speak answers back to you**. "
                "Type or use 🎙 below to ask a question — "
                "I'll answer from your PDFs and read it aloud.\n\n"
                "**Try asking:**\n"
                "- *Explain Newton's interpolation formula*\n"
                "- *What is CPU scheduling?*\n"
                "- *Summarise chapter 1*"
            )
        else:
            st.markdown(
                "👋 Hi! I'm **NoteNest**.\n\n"
                "📂 Upload your lecture PDFs in the sidebar → click **Ingest**."
            )

# ── Chat history ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Re-render audio player for past assistant messages
        if (message["role"] == "assistant"
                and "audio_b64" in message
                and message["audio_b64"]):
            st.markdown('<div class="audio-label">🔊 Replay answer</div>',
                        unsafe_allow_html=True)
            audio_bytes = base64.b64decode(message["audio_b64"])
            st.audio(audio_bytes, format="audio/mp3")

        # Sources
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
# VOICE INPUT
# ─────────────────────────────────────────────
if voice_input_ok and st.session_state.kb_ready:
    st.markdown("---")
    st.markdown("**🎙️ Voice Input**")

    col_rec, col_status = st.columns([1, 3])
    with col_rec:
        audio_data = st.audio_input(
            "Click to record",
            label_visibility="collapsed",
            key="audio_recorder"
        )
    with col_status:
        if audio_data:
            st.markdown(
                '<div class="voice-card">🎧 Transcribing...</div>',
                unsafe_allow_html=True
            )

    if audio_data:
        with st.spinner("Transcribing with Whisper..."):
            try:
                transcript = transcribe_audio_file(
                    audio_data.read(),
                    use_api=use_api_whisper
                )
                if transcript:
                    st.session_state.voice_transcript = transcript
                    st.success(f"🎤 Heard: *\"{transcript}\"*")
                else:
                    st.warning("Couldn't hear clearly. Try again.")
            except Exception as e:
                st.error(f"Transcription error: {e}")


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
prefill = st.session_state.get("voice_transcript", "")

question = st.chat_input(
    "Ask a question... (or use 🎙 above to speak)",
    disabled=not st.session_state.kb_ready,
)

# Use voice transcript if no typed question
if not question and prefill:
    question = prefill
    st.session_state.voice_transcript = ""

# ── Process question ──
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ── Generate answer ──
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                chain = get_chain()
                result = chain({
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                })
                answer      = result["answer"]
                source_docs = result.get("source_documents", [])
            except Exception as e:
                answer      = f"Something went wrong: {e}"
                source_docs = []

        st.markdown(answer)

        # ── Voice output ──
        audio_b64 = None
        if voice_responses and voice_output_ok:
            with st.spinner("🔊 Generating audio..."):
                try:
                    audio_b64 = speak(
                        answer,
                        lang=tts_lang_code,
                        force_gtts=(tts_status["active_engine"] != "ElevenLabs"),
                        elevenlabs_voice=selected_voice or "Rachel (calm)",
                    )
                except Exception as e:
                    st.warning(f"TTS error: {e}")

            if audio_b64:
                # Auto-play
                autoplay_audio(audio_b64)

                # Also show manual replay player
                st.markdown('<div class="audio-label">🔊 Answer audio</div>',
                            unsafe_allow_html=True)
                audio_bytes = base64.b64decode(audio_b64)
                st.audio(audio_bytes, format="audio/mp3")

        # ── Sources ──
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

    # Save message with audio
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "sources":   sources,
        "audio_b64": audio_b64,
    })

    st.session_state.chat_history.append((question, answer))
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]

    st.session_state.voice_transcript = ""
