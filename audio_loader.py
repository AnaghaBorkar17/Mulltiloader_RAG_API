# app/audio_loader.py

from pathlib import Path
from langchain_core.documents import Document
import numpy as np

_whisper_model = None


def get_whisper_model():
    """
    Lazy-load the lightweight Whisper model.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Please install it with: pip install openai-whisper"
            )
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def load_audio_file_pure_python(file_path: str):
    """
    Read WAV audio directly using scipy without requiring ffmpeg.exe.
    """
    from scipy.io import wavfile
    from scipy.signal import resample

    sample_rate, data = wavfile.read(file_path)

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to float32 between -1.0 and 1.0
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)

    # Resample to 16,000 Hz for Whisper
    target_sr = 16000
    if sample_rate != target_sr:
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = resample(audio, num_samples).astype(np.float32)

    return audio


def load_audio(file_path: str):
    """
    Transcribe audio files (WAV, MP3, M4A) into text using local Whisper.
    Supports pure-Python WAV decoding without external ffmpeg.
    """
    model = get_whisper_model()
    ext = Path(file_path).suffix.lower()

    if ext == ".wav":
        try:
            audio_array = load_audio_file_pure_python(file_path)
            result = model.transcribe(audio_array, fp16=False)
        except Exception:
            # Fallback to standard transcribe
            result = model.transcribe(file_path, fp16=False)
    else:
        result = model.transcribe(file_path, fp16=False)

    text = result.get("text", "").strip()

    if not text:
        raise ValueError("No speech could be detected in audio.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "file_type": "audio"
            }
        )
    ]
