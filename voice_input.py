"""
NoteNest — Day 4
voice_input.py — Voice Recording + Whisper Transcription

This module handles:
  1. Recording audio from the microphone (sounddevice)
  2. Saving to a temporary WAV file
  3. Transcribing with OpenAI Whisper (local, free)
  4. Returning the transcribed text to app.py

Two transcription modes:
  - Local Whisper (free, offline, ~1 GB model download on first use)
  - Whisper API  (paid, faster, no download needed)

Usage:
    from voice_input import transcribe_audio_file, record_and_transcribe
"""

import os
import wave
import tempfile
import threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Audio recording configuration
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000   # Whisper works best at 16kHz
CHANNELS        = 1       # Mono audio
DTYPE           = np.int16
CHUNK_DURATION  = 0.1     # seconds per audio chunk
WHISPER_MODEL   = "base"  # tiny/base/small/medium — base is best speed/accuracy tradeoff


def check_dependencies() -> dict:
    """
    Check which audio/transcription packages are available.
    Returns a dict so app.py knows what features to enable.
    """
    status = {
        "sounddevice": False,
        "whisper_local": False,
        "whisper_api": False,
    }

    try:
        import sounddevice
        status["sounddevice"] = True
    except ImportError:
        pass

    try:
        import whisper
        status["whisper_local"] = True
    except ImportError:
        pass

    if os.getenv("OPENAI_API_KEY"):
        status["whisper_api"] = True

    return status


# ─────────────────────────────────────────────
# Audio recording
# ─────────────────────────────────────────────
class AudioRecorder:
    """
    Records microphone audio in a background thread.
    Usage:
        recorder = AudioRecorder()
        recorder.start()
        # ... user speaks ...
        recorder.stop()
        wav_path = recorder.save_wav()
    """

    def __init__(self):
        self.frames = []
        self.is_recording = False
        self._thread = None
        self._stream = None

    def start(self):
        """Start recording in background thread."""
        import sounddevice as sd

        self.frames = []
        self.is_recording = True

        def record_loop():
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            ) as stream:
                self._stream = stream
                while self.is_recording:
                    data, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
                    self.frames.append(data.copy())

        self._thread = threading.Thread(target=record_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop recording and wait for thread to finish."""
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def save_wav(self, output_path: str = None) -> str:
        """
        Save recorded audio as a WAV file.
        Returns the path to the saved file.
        """
        if not self.frames:
            raise ValueError("No audio recorded. Did you call start() first?")

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        audio_data = np.concatenate(self.frames, axis=0)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)   # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        duration = len(audio_data) / SAMPLE_RATE
        print(f"🎤  Saved {duration:.1f}s of audio to {output_path}")
        return output_path

    def get_duration(self) -> float:
        """Returns current recording duration in seconds."""
        if not self.frames:
            return 0.0
        total_samples = sum(len(f) for f in self.frames)
        return total_samples / SAMPLE_RATE


# ─────────────────────────────────────────────
# Transcription — Local Whisper
# ─────────────────────────────────────────────
def transcribe_local(wav_path: str, model_size: str = WHISPER_MODEL) -> str:
    """
    Transcribe audio using local OpenAI Whisper model.

    Model sizes and tradeoffs:
      tiny   — fastest, least accurate (~75 MB)
      base   — good balance, recommended (~150 MB)
      small  — more accurate, slower (~500 MB)
      medium — best accuracy, requires more RAM (~1.5 GB)

    First call downloads the model automatically.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Whisper not installed.\n"
            "Run: pip install openai-whisper"
        )

    print(f"🎧  Transcribing with local Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    result = model.transcribe(
        wav_path,
        language=None,          # auto-detect language
        fp16=False,             # use fp32 for CPU compatibility
        verbose=False,
    )

    text = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    print(f"✅  Transcribed ({detected_lang}): '{text}'")
    return text


# ─────────────────────────────────────────────
# Transcription — Whisper API (paid but fast)
# ─────────────────────────────────────────────
def transcribe_api(wav_path: str) -> str:
    """
    Transcribe using OpenAI Whisper API.
    Faster than local but costs ~$0.006 per minute.
    Requires OPENAI_API_KEY in .env
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")

    print("🎧  Transcribing with Whisper API...")
    client = OpenAI(api_key=api_key)

    with open(wav_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    text = response.strip()
    print(f"✅  Transcribed (API): '{text}'")
    return text


# ─────────────────────────────────────────────
# Transcribe from uploaded file bytes (Streamlit)
# ─────────────────────────────────────────────
def transcribe_audio_file(audio_bytes: bytes, use_api: bool = False) -> str:
    """
    Transcribe audio from raw bytes (from Streamlit's audio_input or file_uploader).

    Args:
        audio_bytes: Raw audio bytes
        use_api: True = Whisper API (paid), False = local Whisper (free)

    Returns:
        Transcribed text string
    """
    # Save bytes to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if use_api:
            return transcribe_api(tmp_path)
        else:
            return transcribe_local(tmp_path)
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("NoteNest Voice Input Test")
    print("=" * 40)

    deps = check_dependencies()
    print(f"sounddevice : {'✅' if deps['sounddevice'] else '❌ pip install sounddevice'}")
    print(f"whisper     : {'✅' if deps['whisper_local'] else '❌ pip install openai-whisper'}")
    print(f"whisper api : {'✅' if deps['whisper_api'] else '⚠️  No OPENAI_API_KEY'}")

    if deps["sounddevice"] and deps["whisper_local"]:
        print("\n🎤  Recording for 5 seconds... speak now!")
        recorder = AudioRecorder()
        recorder.start()
        time.sleep(5)
        recorder.stop()
        wav_path = recorder.save_wav()
        text = transcribe_local(wav_path)
        print(f"\n📝  You said: '{text}'")
        Path(wav_path).unlink(missing_ok=True)
    else:
        print("\n⚠️  Install missing packages above then retry.")
