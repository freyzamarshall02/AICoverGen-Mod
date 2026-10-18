import librosa
import soxr
import numpy as np


def load_audio(file, sr):
    """
    Load an audio file and resample it to the target sample rate.
    Uses librosa for decoding and soxr for high-quality resampling.

    Args:
        file (str): Path to the audio file.
        sr (int): Target sample rate.

    Returns:
        np.ndarray: Mono float32 audio array at the target sample rate.
    """
    file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    try:
        audio, orig_sr = librosa.load(file, sr=None, mono=True)
        if orig_sr != sr:
            audio = soxr.resample(audio, orig_sr, sr, quality='VHQ')
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return audio.astype(np.float32)
