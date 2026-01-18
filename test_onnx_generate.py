#!/usr/bin/env python3
"""Simple ONNX TTS test using the exported Piper model."""

import argparse
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.piper.phoneme_ids import phonemes_to_ids
from src.piper.phonemize_espeak import EspeakPhonemizer


def text_to_ids(text: str, voice: str, phoneme_id_map: dict[str, list[int]]) -> list[int]:
    """Phonemize text with espeak-ng and convert phonemes to id sequence using model's phoneme map."""
    phonemizer = EspeakPhonemizer()
    sentences = phonemizer.phonemize(voice, text)
    if not sentences:
        raise SystemExit("Phonemizer returned no phonemes")

    # Flatten sentences; insert a space between sentences to keep pause
    flat_phonemes: list[str] = []
    for idx, sentence in enumerate(sentences):
        flat_phonemes.extend(sentence)
        if idx < len(sentences) - 1:
            flat_phonemes.append(" ")

    return phonemes_to_ids(flat_phonemes, phoneme_id_map)


def run_inference(model_path: Path, ids: list[int], noise: float, length: float, noise_w: float) -> np.ndarray:
    """Run ONNX model and return audio as float32 numpy array."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    sequences = np.array(ids, dtype=np.int64)[None, :]
    sequence_lengths = np.array([sequences.shape[1]], dtype=np.int64)
    scales = np.array([noise, length, noise_w], dtype=np.float32)

    output = sess.run(
        None,
        {
            "input": sequences,
            "input_lengths": sequence_lengths,
            "scales": scales,
        },
    )[0]

    return np.squeeze(output)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save mono audio to WAV (16-bit PCM)."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ONNX voice generation")
    parser.add_argument("--model", default="felix_mirage_epoch426.onnx", help="Path to ONNX model")
    parser.add_argument("--config", help="Path to model config JSON (defaults to MODEL.onnx.json)")
    parser.add_argument("--voice", default="ru", help="espeak-ng voice code (e.g., ru)")
    parser.add_argument("--text", default="Привет! Это тест голоса.", help="Text to synthesize")
    parser.add_argument("--output", default="test_output.wav", help="Where to save WAV")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Output sample rate")
    parser.add_argument("--noise", type=float, default=0.667, help="Noise scale")
    parser.add_argument("--length", type=float, default=1.0, help="Length scale")
    parser.add_argument("--noise-w", type=float, default=0.8, help="Noise scale for duration predictor")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    config_path = Path(args.config) if args.config else model_path.with_suffix(model_path.suffix + ".json")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    phoneme_id_map = config.get("phoneme_id_map")
    if not phoneme_id_map:
        raise SystemExit("phoneme_id_map not found in config")

    ids = text_to_ids(args.text, args.voice, phoneme_id_map)
    audio = run_inference(model_path, ids, args.noise, args.length, args.noise_w)
    save_wav(Path(args.output), audio, args.sample_rate)

    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()
