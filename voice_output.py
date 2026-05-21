"""
NoteNest — Day 5
voice_output.py — Text-to-Speech Output

Two TTS engines supported:
  1. gTTS  (Google Text-to-Speech) — FREE, no API key, good quality
  2. ElevenLabs — Premium, ultra-realistic voices, needs API key

Auto-selects ElevenLabs if ELEVENLABS_API_KEY is in .env,
falls back to gTTS otherwise.

Usage:
    from voice_output import speak, get_tts_status
    audio_b64 = speak("Hello, here is your answer.")
    # Returns base64 encoded MP3 for Streamlit audio playback
"""

import os
import base64
import tempfile
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Supported languages for gTTS
# ─────────────────────────────────────────────
GTTS_LANG_MAP = {
    "English":    "en",
    "Hindi":      "hi",
    "Bengali":    "bn",
    "Spanish":    "es",
    "French":     "fr",
    "Arabic":     "ar",
    "German":     "de",
    "Japanese":   "ja",
    "Chinese":    "zh",
    "Portuguese": "pt",
}

# ElevenLabs voice IDs (free tier voices)
ELEVENLABS_VOICES = {
    "Rachel (calm)":   "21m00Tcm4TlvDq8ikWAM",
    "Domi (confident)":"AZnzlk1XvdvUeBnXmlld",
    "Bella (soft)":    "EXAVITQu4vr4xnSDxMaL",
    "Antoni (warm)":   "ErXwobaYiN019PkySvjV",
    "Josh (deep)":     "TxGEqnHWrfWFTfGW9XjX",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clean_text_for_speech(text: str) -> str:
    """
    Strip markdown, LaTeX, and special chars that sound
    bad when read aloud.
    """
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove LaTeX inline math — replace with "formula"
    text = re.sub(r'\$\$?.+?\$\$?', 'formula', text, flags=re.DOTALL)
    # Remove backtick code
    text = re.sub(r'`{1,3}.+?`{1,3}', '', text, flags=re.DOTALL)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove bullet characters
    text = re.sub(r'^[\-\*\•]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def trim_for_speech(text: str, max_chars: int = 800) -> str:
    """
    Trim answer to max_chars for TTS — avoids 3-minute
    audio clips for long answers. Cuts at sentence boundary.
    """
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    last_period = trimmed.rfind('.')
    if last_period > max_chars // 2:
        trimmed = trimmed[:last_period + 1]
    return trimmed + " ... (answer trimmed for audio)"


def audio_to_base64(audio_path: str) -> str:
    """Convert audio file to base64 string for Streamlit."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# gTTS — Free Google TTS
# ─────────────────────────────────────────────
def speak_gtts(text: str, lang: str = "en") -> str:
    """
    Convert text to speech using gTTS (free, no API key).

    Args:
        text: Text to speak
        lang: Language code (e.g. 'en', 'hi', 'bn')

    Returns:
        Base64 encoded MP3 string
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError(
            "gTTS not installed.\n"
            "Run: pip install gTTS"
        )

    # Clean and trim text
    clean = clean_text_for_speech(text)
    trimmed = trim_for_speech(clean)

    print(f"🔊  gTTS generating audio ({lang}, {len(trimmed)} chars)...")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts = gTTS(text=trimmed, lang=lang, slow=False)
        tts.save(tmp_path)
        audio_b64 = audio_to_base64(tmp_path)
        print("✅  gTTS audio ready")
        return audio_b64
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# ElevenLabs — Premium ultra-realistic TTS
# ─────────────────────────────────────────────
def speak_elevenlabs(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model: str = "eleven_turbo_v2"
) -> str:
    """
    Convert text to speech using ElevenLabs API.
    Ultra-realistic voices — great for demo videos.

    Requires ELEVENLABS_API_KEY in .env
    Free tier: 10,000 chars/month

    Args:
        text    : Text to speak
        voice_id: ElevenLabs voice ID
        model   : eleven_turbo_v2 (fast) or eleven_multilingual_v2

    Returns:
        Base64 encoded MP3 string
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ELEVENLABS_API_KEY not found in .env\n"
            "Get free key at: elevenlabs.io\n"
            "Falling back to gTTS."
        )

    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
    except ImportError:
        raise ImportError(
            "ElevenLabs not installed.\n"
            "Run: pip install elevenlabs"
        )

    clean = clean_text_for_speech(text)
    trimmed = trim_for_speech(clean, max_chars=1000)

    print(f"🔊  ElevenLabs generating audio ({len(trimmed)} chars)...")

    client = ElevenLabs(api_key=api_key)

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=trimmed,
        model_id=model,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        for chunk in audio_generator:
            tmp.write(chunk)

    try:
        audio_b64 = audio_to_base64(tmp_path)
        print("✅  ElevenLabs audio ready")
        return audio_b64
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Main speak function — auto-selects engine
# ─────────────────────────────────────────────
def speak(
    text: str,
    lang: str = "en",
    force_gtts: bool = False,
    elevenlabs_voice: str = "Rachel (calm)",
) -> str | None:
    """
    Convert text to speech. Auto-selects ElevenLabs if
    API key is present, otherwise uses gTTS.

    Args:
        text              : Answer text to speak
        lang              : Language code for gTTS
        force_gtts        : Force gTTS even if ElevenLabs key exists
        elevenlabs_voice  : Voice name from ELEVENLABS_VOICES dict

    Returns:
        Base64 MP3 string, or None if TTS unavailable
    """
    if not text or not text.strip():
        return None

    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    if elevenlabs_key and not force_gtts:
        try:
            voice_id = ELEVENLABS_VOICES.get(
                elevenlabs_voice,
                "21m00Tcm4TlvDq8ikWAM"
            )
            return speak_elevenlabs(text, voice_id=voice_id)
        except Exception as e:
            print(f"⚠️  ElevenLabs failed ({e}), falling back to gTTS")

    # gTTS fallback
    try:
        return speak_gtts(text, lang=lang)
    except Exception as e:
        print(f"❌  gTTS also failed: {e}")
        return None


# ─────────────────────────────────────────────
# Status check
# ─────────────────────────────────────────────
def get_tts_status() -> dict:
    """Returns which TTS engines are available."""
    status = {
        "gtts": False,
        "elevenlabs": False,
        "active_engine": "none",
    }

    try:
        from gtts import gTTS
        status["gtts"] = True
    except ImportError:
        pass

    if os.getenv("ELEVENLABS_API_KEY"):
        try:
            from elevenlabs.client import ElevenLabs
            status["elevenlabs"] = True
        except ImportError:
            pass

    if status["elevenlabs"]:
        status["active_engine"] = "ElevenLabs"
    elif status["gtts"]:
        status["active_engine"] = "gTTS"

    return status


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("NoteNest — Voice Output Test")
    print("=" * 40)

    status = get_tts_status()
    print(f"gTTS        : {'✅' if status['gtts'] else '❌ pip install gTTS'}")
    print(f"ElevenLabs  : {'✅' if status['elevenlabs'] else '⚠️  No ELEVENLABS_API_KEY'}")
    print(f"Active engine: {status['active_engine']}")

    if status["gtts"] or status["elevenlabs"]:
        print("\n🔊  Generating test audio...")
        sample = (
            "Hello! I am NoteNest, your AI study assistant. "
            "I can now speak answers from your lecture notes."
        )
        b64 = speak(sample)
        if b64:
            # Save to test file
            with open("test_audio.mp3", "wb") as f:
                f.write(base64.b64decode(b64))
            print("✅  Saved test_audio.mp3 — open it to hear NoteNest speak!")
        else:
            print("❌  Audio generation failed.")
