#!/usr/bin/env python3
"""Dataset quality report (fast pre-Whisper gate).

Generates a human-reviewable report before running expensive full Whisper alignment.
Local-only; does not use paid APIs.

Artifacts written under --report-dir:
- quality_report.md
- samples.tsv
- suspects.tsv
- stats.json
- (optional) asr_sample.tsv

This script is intentionally dependency-light.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio() if a or b else 1.0


@dataclass
class Row:
    row_num: int
    wav: str
    text: str


@dataclass
class AudioInfo:
    duration_s: float | None
    samplerate: int | None
    channels: int | None
    subtype: str | None
    error: str | None


def read_metadata(metadata_path: Path) -> list[Row]:
    rows: list[Row] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for i, parts in enumerate(reader, start=1):
            if not parts or all(not p.strip() for p in parts):
                rows.append(Row(i, "", ""))
                continue
            wav = (parts[0] or "").strip()
            text = (parts[1] if len(parts) > 1 else "").strip()
            rows.append(Row(i, wav, text))
    return rows


def get_audio_info(wav_path: Path) -> AudioInfo:
    if not wav_path.exists():
        return AudioInfo(None, None, None, None, "missing_wav")

    if sf is not None:
        try:
            info = sf.info(str(wav_path))
            duration = (info.frames / info.samplerate) if info.samplerate else None
            return AudioInfo(duration, info.samplerate, info.channels, info.subtype, None)
        except Exception:
            return AudioInfo(None, None, None, None, "invalid_audio")

    # Fallback: wave (works for PCM WAV)
    try:
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            frames = w.getnframes()
            ch = w.getnchannels()
            duration = (frames / sr) if sr else None
        return AudioInfo(duration, sr, ch, "unknown", None)
    except Exception:
        return AudioInfo(None, None, None, None, "invalid_audio")


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _ascii_hist(values: list[float], *, bins: list[float]) -> list[str]:
    if not values:
        return ["(no data)"]
    counts = [0 for _ in range(len(bins) - 1)]
    for v in values:
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if (v >= lo) and (v < hi or (i == len(bins) - 2 and v <= hi)):
                counts[i] += 1
                break
    maxc = max(counts) if counts else 1
    lines: list[str] = []
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i + 1]
        bar = "#" * int((c / maxc) * 40) if maxc > 0 else ""
        lines.append(f"{lo:>5.1f}..{hi:>5.1f} | {c:>6} {bar}")
    return lines


def _token_repetition_score(text: str) -> float:
    norm = normalize_text(text)
    if not norm:
        return 0.0
    tokens = norm.split()
    if len(tokens) < 6:
        return 0.0
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    most = max(counts.values())
    return most / max(1, len(tokens))


def _select_whisper_backend(preferred: str) -> str:
    if preferred in {"whisper", "faster-whisper"}:
        return preferred
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return "faster-whisper"
    except Exception:
        return "whisper"


def _load_whisper_model(backend: str, model_name: str, device: str, compute_type: str):
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel(model_name, device=device, compute_type=compute_type)
    if backend == "whisper":
        import whisper  # type: ignore

        return whisper.load_model(model_name, device=device)
    raise RuntimeError(f"Unknown whisper backend: {backend}")


def _whisper_transcribe(
    wav_path: Path,
    *,
    backend: str,
    model: Any,
    device: str,
    language: str | None,
    batch_size: int,
    num_workers: int,
    beam_size: int,
    vad_filter: bool,
) -> str:
    if backend == "faster-whisper":
        kwargs: dict[str, Any] = {
            "beam_size": beam_size,
            "language": language,
            "vad_filter": vad_filter,
        }
        if batch_size > 0:
            kwargs["batch_size"] = batch_size
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
        try:
            segments, _ = model.transcribe(str(wav_path), **kwargs)
        except TypeError:
            kwargs.pop("batch_size", None)
            kwargs.pop("num_workers", None)
            segments, _ = model.transcribe(str(wav_path), **kwargs)
        return " ".join(seg.text for seg in segments).strip()

    if backend == "whisper":
        result = model.transcribe(
            str(wav_path),
            fp16=(device == "cuda"),
            language=language,
        )
        return (result.get("text") or "").strip()

    raise RuntimeError("Whisper backend not available")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset root (config.json, metadata_2col.csv, wavs/)")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Output directory (default: <dataset>/reports/quality_<UTC_TIMESTAMP>)",
    )
    parser.add_argument("--sample-count", type=int, default=30, help="How many random clips to list for listening")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 = time-based)")
    parser.add_argument("--suspect-top", type=int, default=200, help="Max suspect rows to list in suspects.tsv")

    # heuristics (keep aligned with validate_dataset_full defaults)
    parser.add_argument("--min-text-len", type=int, default=3)
    parser.add_argument("--max-text-len", type=int, default=400)
    parser.add_argument("--min-cyrillic-ratio", type=float, default=0.6)
    parser.add_argument("--min-duration", type=float, default=0.2)
    parser.add_argument("--max-duration", type=float, default=15.5)
    parser.add_argument("--latin-suspect-ratio", type=float, default=0.1)
    parser.add_argument("--repetition-suspect", type=float, default=0.5, help="Token repetition score threshold")

    # optional quick ASR sample
    parser.add_argument("--asr-sample", type=int, default=0, help="If >0, run ASR on N random clips")
    parser.add_argument("--asr-model", default="medium", help="Whisper model for quick sample (default: medium)")
    parser.add_argument("--asr-device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--asr-backend",
        choices=["auto", "faster-whisper", "whisper"],
        default="auto",
        help="Whisper backend preference",
    )
    parser.add_argument("--asr-language", default=None, help="Force language (e.g. ru)")
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--asr-num-workers", type=int, default=2)
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument("--asr-vad-filter", action="store_true")
    parser.add_argument("--asr-similarity-threshold", type=float, default=0.8)

    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    metadata_path = dataset / "metadata_2col.csv"
    wavs_dir = dataset / "wavs"

    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata_2col.csv: {metadata_path}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"Missing wavs/: {wavs_dir}")

    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else (dataset / "reports" / f"quality_{_utc_ts()}")
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed != 0 else int(time.time())
    rng = random.Random(seed)

    rows = read_metadata(metadata_path)

    # Collect signals
    durations: list[float] = []
    text_lens: list[int] = []

    normalized_text_to_first: dict[str, int] = {}
    duplicate_rows: list[int] = []

    suspects: list[dict[str, Any]] = []

    # Precompute audio info and basic heuristics
    for r in rows:
        if not r.wav or not r.text:
            suspects.append({"row": r.row_num, "wav": r.wav, "issue": "empty_wav_or_text", "details": "", "text": r.text})
            continue

        if NON_PRINTABLE_RE.search(r.text):
            suspects.append({"row": r.row_num, "wav": r.wav, "issue": "non_printable", "details": "", "text": r.text})

        if len(r.text) < args.min_text_len:
            suspects.append({"row": r.row_num, "wav": r.wav, "issue": "too_short_text", "details": str(len(r.text)), "text": r.text})
        if len(r.text) > args.max_text_len:
            suspects.append({"row": r.row_num, "wav": r.wav, "issue": "too_long_text", "details": str(len(r.text)), "text": r.text})

        letters = len(CYRILLIC_RE.findall(r.text)) + len(LATIN_RE.findall(r.text))
        cyr = len(CYRILLIC_RE.findall(r.text))
        lat = len(LATIN_RE.findall(r.text))
        cyr_ratio = (cyr / letters) if letters else 0.0
        lat_ratio = (lat / letters) if letters else 0.0
        if letters and cyr_ratio < args.min_cyrillic_ratio:
            suspects.append(
                {
                    "row": r.row_num,
                    "wav": r.wav,
                    "issue": "low_cyrillic_ratio",
                    "details": f"cyr_ratio={cyr_ratio:.3f}",
                    "text": r.text,
                }
            )
        if letters and (cyr > 0 and lat > 0) and (lat_ratio >= args.latin_suspect_ratio):
            suspects.append(
                {
                    "row": r.row_num,
                    "wav": r.wav,
                    "issue": "code_switch",
                    "details": f"lat_ratio={lat_ratio:.3f}",
                    "text": r.text,
                }
            )

        rep = _token_repetition_score(r.text)
        if rep >= args.repetition_suspect:
            suspects.append(
                {
                    "row": r.row_num,
                    "wav": r.wav,
                    "issue": "high_token_repetition",
                    "details": f"score={rep:.3f}",
                    "text": r.text,
                }
            )

        norm = normalize_text(r.text)
        if norm:
            if norm in normalized_text_to_first:
                duplicate_rows.append(r.row_num)
            else:
                normalized_text_to_first[norm] = r.row_num

        wav_path = wavs_dir / r.wav
        ainfo = get_audio_info(wav_path)
        if ainfo.error:
            suspects.append({"row": r.row_num, "wav": r.wav, "issue": ainfo.error, "details": "", "text": r.text})
            continue
        if ainfo.duration_s is not None:
            durations.append(ainfo.duration_s)
            if ainfo.duration_s < args.min_duration:
                suspects.append(
                    {
                        "row": r.row_num,
                        "wav": r.wav,
                        "issue": "too_short_audio",
                        "details": f"dur={ainfo.duration_s:.3f}",
                        "text": r.text,
                    }
                )
            if ainfo.duration_s > args.max_duration:
                suspects.append(
                    {
                        "row": r.row_num,
                        "wav": r.wav,
                        "issue": "too_long_audio",
                        "details": f"dur={ainfo.duration_s:.3f}",
                        "text": r.text,
                    }
                )

        text_lens.append(len(r.text))

    # Mark duplicates (after first pass)
    for row_num in duplicate_rows:
        suspects.append({"row": row_num, "wav": rows[row_num - 1].wav, "issue": "duplicate_text", "details": "", "text": rows[row_num - 1].text})

    # Samples to listen
    valid_rows = [r for r in rows if r.wav and (wavs_dir / r.wav).is_file()]
    rng.shuffle(valid_rows)
    sample_rows = valid_rows[: max(0, args.sample_count)]

    # Write samples.tsv
    samples_path = report_dir / "samples.tsv"
    with samples_path.open("w", encoding="utf-8", newline="") as f:
        f.write("row\twav\tduration_s\ttext\n")
        for r in sample_rows:
            ainfo = get_audio_info(wavs_dir / r.wav)
            dur = ("" if ainfo.duration_s is None else f"{ainfo.duration_s:.3f}")
            f.write(f"{r.row_num}\t{r.wav}\t{dur}\t{r.text}\n")

    # Optional ASR sample
    asr_results: list[dict[str, Any]] = []
    asr_low_sim: list[dict[str, Any]] = []
    asr_backend = None
    if args.asr_sample and args.asr_sample > 0:
        asr_backend = _select_whisper_backend(args.asr_backend)
        model = _load_whisper_model(asr_backend, args.asr_model, args.asr_device, args.asr_compute_type)

        asr_rows = valid_rows.copy()
        rng.shuffle(asr_rows)
        asr_rows = asr_rows[: min(args.asr_sample, len(asr_rows))]

        for i, r in enumerate(asr_rows, start=1):
            wav_path = wavs_dir / r.wav
            try:
                hyp = _whisper_transcribe(
                    wav_path,
                    backend=asr_backend,
                    model=model,
                    device=args.asr_device,
                    language=args.asr_language,
                    batch_size=args.asr_batch_size,
                    num_workers=args.asr_num_workers,
                    beam_size=args.asr_beam_size,
                    vad_filter=args.asr_vad_filter,
                )
                sim = compute_similarity(normalize_text(r.text), normalize_text(hyp))
                rec = {
                    "row": r.row_num,
                    "wav": r.wav,
                    "ref": r.text,
                    "hyp": hyp,
                    "sim": round(float(sim), 4),
                }
                asr_results.append(rec)
                if sim < args.asr_similarity_threshold:
                    asr_low_sim.append(rec)
                    suspects.append(
                        {
                            "row": r.row_num,
                            "wav": r.wav,
                            "issue": "asr_low_similarity",
                            "details": f"sim={sim:.3f}",
                            "text": r.text,
                        }
                    )
            except Exception as exc:
                rec = {"row": r.row_num, "wav": r.wav, "ref": r.text, "hyp": "", "sim": None, "error": str(exc)}
                asr_results.append(rec)
                suspects.append({"row": r.row_num, "wav": r.wav, "issue": "asr_error", "details": str(exc), "text": r.text})

            if i % 25 == 0:
                print(f"[asr] {i}/{len(asr_rows)} processed", flush=True)

        asr_path = report_dir / "asr_sample.tsv"
        with asr_path.open("w", encoding="utf-8", newline="") as f:
            f.write("row\twav\tsim\tref\thyp\terror\n")
            for rec in asr_results:
                f.write(
                    "\t".join(
                        [
                            str(rec.get("row")),
                            str(rec.get("wav")),
                            "" if rec.get("sim") is None else str(rec.get("sim")),
                            (rec.get("ref") or "").replace("\t", " "),
                            (rec.get("hyp") or "").replace("\t", " "),
                            (rec.get("error") or "").replace("\t", " "),
                        ]
                    )
                    + "\n"
                )

    # Summaries
    dur_sorted = sorted(durations)
    text_sorted = sorted(text_lens)

    stats: dict[str, Any] = {
        "dataset": str(dataset),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "rows_total": len(rows),
        "rows_with_wav": len(valid_rows),
        "duration": {
            "count": len(durations),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "mean": (statistics.mean(durations) if durations else None),
            "p50": _percentile(dur_sorted, 50),
            "p95": _percentile(dur_sorted, 95),
        },
        "text_len": {
            "count": len(text_lens),
            "min": min(text_lens) if text_lens else None,
            "max": max(text_lens) if text_lens else None,
            "mean": (statistics.mean(text_lens) if text_lens else None),
            "p50": _percentile([float(x) for x in text_sorted], 50),
            "p95": _percentile([float(x) for x in text_sorted], 95),
        },
        "duplicate_text_rows": len(duplicate_rows),
        "suspects_total": len(suspects),
        "asr": {
            "ran": bool(args.asr_sample and args.asr_sample > 0),
            "backend": asr_backend,
            "model": args.asr_model if args.asr_sample and args.asr_sample > 0 else None,
            "device": args.asr_device if args.asr_sample and args.asr_sample > 0 else None,
            "sample_n": int(args.asr_sample) if args.asr_sample and args.asr_sample > 0 else 0,
            "low_similarity_n": len(asr_low_sim),
        },
    }

    (report_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write suspects.tsv
    suspects_path = report_dir / "suspects.tsv"
    # Group to keep the file readable
    suspects_sorted = sorted(suspects, key=lambda d: (str(d.get("issue")), int(d.get("row") or 0)))
    suspects_sorted = suspects_sorted[: max(0, args.suspect_top)]
    with suspects_path.open("w", encoding="utf-8", newline="") as f:
        f.write("row\twav\tissue\tdetails\ttext\n")
        for s in suspects_sorted:
            text = (s.get("text") or "").replace("\t", " ")
            f.write(
                f"{s.get('row')}\t{s.get('wav')}\t{s.get('issue')}\t{s.get('details','')}\t{text}\n"
            )

    # Build Markdown report
    md_lines: list[str] = []
    md_lines.append("# Dataset Quality Report\n")
    md_lines.append(f"**Dataset:** {dataset}\n")
    md_lines.append(f"**Date (UTC):** {datetime.now(timezone.utc).isoformat()}\n")
    md_lines.append(f"**Seed:** {seed}\n")

    md_lines.append("\n## Summary\n")
    md_lines.append(f"- Total metadata rows: {len(rows)}\n")
    md_lines.append(f"- Rows with existing wav: {len(valid_rows)}\n")
    md_lines.append(f"- Suspect signals (raw): {len(suspects)}\n")
    md_lines.append(f"- Duplicate normalized texts: {len(duplicate_rows)}\n")
    if args.asr_sample and args.asr_sample > 0:
        md_lines.append(f"- Quick ASR sample: {int(args.asr_sample)} clips ({asr_backend}, model={args.asr_model}, device={args.asr_device})\n")
        md_lines.append(f"- ASR low similarity (<{args.asr_similarity_threshold}): {len(asr_low_sim)}\n")
    else:
        md_lines.append("- Quick ASR sample: not run\n")

    md_lines.append("\n## What to Review\n")
    md_lines.append(f"- Listening list: {samples_path.name}\n")
    md_lines.append(f"- Suspect rows list: {suspects_path.name}\n")
    md_lines.append("\nSuggested workflow:\n")
    md_lines.append("1. Listen to samples.tsv items (10–50).\n")
    md_lines.append("2. Spot-check suspects.tsv issues (especially missing/invalid audio, code-switch, duplicates).\n")
    md_lines.append("3. If clean: proceed to full sharded Whisper validation (Stage 3).\n")

    md_lines.append("\n## Duration Statistics (seconds)\n")
    md_lines.append(f"- min={stats['duration']['min']} max={stats['duration']['max']} mean={stats['duration']['mean']} p50={stats['duration']['p50']} p95={stats['duration']['p95']}\n")
    md_lines.append("\nHistogram:\n\n```")
    md_lines.extend(_ascii_hist(durations, bins=[0, 0.5, 1, 2, 3, 5, 8, 12, 15, 20]))
    md_lines.append("```\n")

    md_lines.append("\n## Text Length Statistics (chars)\n")
    md_lines.append(f"- min={stats['text_len']['min']} max={stats['text_len']['max']} mean={stats['text_len']['mean']} p50={stats['text_len']['p50']} p95={stats['text_len']['p95']}\n")

    (report_dir / "quality_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Report dir: {report_dir}")
    print(f"- quality_report.md")
    print(f"- samples.tsv")
    print(f"- suspects.tsv")
    if args.asr_sample and args.asr_sample > 0:
        print(f"- asr_sample.tsv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
