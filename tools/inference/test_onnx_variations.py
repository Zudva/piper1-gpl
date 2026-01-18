#!/usr/bin/env python3
"""Generate multiple audio samples with different parameter variations."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import wave

from src.piper.phoneme_ids import phonemes_to_ids
from src.piper.phonemize_espeak import EspeakPhonemizer


# Предустановки параметров для тестирования
PRESETS = {
    "default": {"noise": 0.667, "length": 1.0, "noise_w": 0.8},
    "clean": {"noise": 0.3, "length": 1.0, "noise_w": 0.5},
    "natural": {"noise": 0.5, "length": 1.0, "noise_w": 0.7},
    "fast": {"noise": 0.667, "length": 0.85, "noise_w": 0.8},
    "slow": {"noise": 0.667, "length": 1.15, "noise_w": 0.8},
    "expressive": {"noise": 0.8, "length": 1.05, "noise_w": 0.9},
    "stable": {"noise": 0.4, "length": 0.95, "noise_w": 0.6},
    "smooth": {"noise": 0.35, "length": 1.0, "noise_w": 0.4},
}


def text_to_ids(text: str, voice: str, phoneme_id_map: dict) -> list[int]:
    """Phonemize text with espeak-ng and convert phonemes to id sequence."""
    phonemizer = EspeakPhonemizer()
    sentences = phonemizer.phonemize(voice, text)
    if not sentences:
        raise SystemExit("Phonemizer returned no phonemes")

    flat_phonemes: list[str] = []
    for idx, sentence in enumerate(sentences):
        flat_phonemes.extend(sentence)
        if idx < len(sentences) - 1:
            flat_phonemes.append(" ")

    return phonemes_to_ids(flat_phonemes, phoneme_id_map)


def run_inference(
    model_path: Path, ids: list[int], noise: float, length: float, noise_w: float
) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Generate multiple variations")
    parser.add_argument(
        "--model", default="felix_mirage_epoch426.onnx", help="Path to ONNX model"
    )
    parser.add_argument(
        "--config", help="Path to model config JSON (defaults to MODEL.json)"
    )
    parser.add_argument("--voice", default="ru", help="espeak-ng voice code")
    parser.add_argument(
        "--text", default="Привет! Это тест голоса.", help="Text to synthesize"
    )
    parser.add_argument("--output-dir", default="test_variations", help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(PRESETS.keys()),
        choices=list(PRESETS.keys()),
        help="Which presets to generate",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    config_path = Path(args.config) if args.config else Path(f"{args.model}.json")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    phoneme_id_map = config.get("phoneme_id_map")
    if not phoneme_id_map:
        raise SystemExit("phoneme_id_map not found in config")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}/")
    print()

    ids = text_to_ids(args.text, args.voice, phoneme_id_map)
    print(f"Phoneme IDs count: {len(ids)}")
    print()

    for preset_name in args.presets:
        params = PRESETS[preset_name]
        output_path = output_dir / f"{preset_name}.wav"

        print(
            f"Generating {preset_name}: noise={params['noise']}, "
            f"length={params['length']}, noise_w={params['noise_w']}"
        )

        audio = run_inference(
            model_path, ids, params["noise"], params["length"], params["noise_w"]
        )
        save_wav(output_path, audio, args.sample_rate)

        duration_sec = len(audio) / args.sample_rate
        print(f"  → {output_path} ({duration_sec:.2f}s)")

    print()
    print(f"Done. Generated {len(args.presets)} variations in {output_dir}/")


if __name__ == "__main__":
    main()
