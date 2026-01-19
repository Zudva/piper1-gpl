#!/usr/bin/env python3
"""
Full dataset validation for Piper training.
Generates TSV + Markdown report with PASS/FAIL verdict.
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio() if a or b else 1.0


def load_config(config_path: Path):
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_metadata(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for i, row in enumerate(reader, start=1):
            if not row or all(not part.strip() for part in row):
                rows.append((i, "", "", "empty_row"))
                continue
            if len(row) < 2:
                rows.append((i, row[0].strip(), "", "missing_text"))
                continue
            wav, text = row[0].strip(), row[1].strip()
            rows.append((i, wav, text, ""))
    return rows


def check_audio_info(wav_path: Path, expected_sr: int | None):
    if sf is None:
        return {"error": "soundfile_not_installed"}
    try:
        info = sf.info(str(wav_path))
    except Exception:
        return {"error": "invalid_audio"}

    issues = []
    if info.channels != 1:
        issues.append("not_mono")
    if info.subtype != "PCM_16":
        issues.append("not_pcm16")
    if expected_sr and info.samplerate != expected_sr:
        issues.append("sample_rate_mismatch")
    duration = info.frames / info.samplerate if info.samplerate else 0
    if duration <= 0.2:
        issues.append("too_short")

    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "duration": duration,
        "issues": issues,
    }


_WHISPER_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


def _load_whisper_model(backend: str, model_name: str, device: str, compute_type: str):
    key = (backend, model_name, device, compute_type)
    if key in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[key]

    if backend == "faster-whisper":
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    elif backend == "whisper":
        import whisper  # type: ignore

        model = whisper.load_model(model_name, device=device)
    else:
        raise RuntimeError(f"Unknown whisper backend: {backend}")

    _WHISPER_MODEL_CACHE[key] = model
    return model


def _select_whisper_backend(preferred: str) -> str:
    if preferred in {"whisper", "faster-whisper"}:
        return preferred
    # auto
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return "faster-whisper"
    except Exception:
        return "whisper"


def whisper_transcribe(
    wav_path: Path,
    *,
    backend: str,
    model,
    device: str,
    language: str | None,
    batch_size: int,
    num_workers: int,
    beam_size: int,
    vad_filter: bool,
):
    if backend == "faster-whisper":
        kwargs = {
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
            # fallback for older faster-whisper versions
            kwargs.pop("batch_size", None)
            kwargs.pop("num_workers", None)
            segments, _ = model.transcribe(str(wav_path), **kwargs)
        text = " ".join(seg.text for seg in segments)
        return text.strip()

    if backend == "whisper":
        result = model.transcribe(
            str(wav_path),
            fp16=(device == "cuda"),
            language=language,
        )
        return (result.get("text") or "").strip()

    raise RuntimeError("Whisper backend not available")


def main():
    parser = argparse.ArgumentParser(description="Full dataset validation (100%)")
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--min-text-len", type=int, default=3)
    parser.add_argument("--max-text-len", type=int, default=400)
    parser.add_argument("--min-cyrillic-ratio", type=float, default=0.6)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--whisper", action="store_true", help="Enable Whisper alignment")
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--whisper-device", default="cuda")
    parser.add_argument("--require-whisper", action="store_true")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument(
        "--whisper-backend",
        choices=["auto", "faster-whisper", "whisper"],
        default="auto",
        help="Whisper backend preference.",
    )
    parser.add_argument("--whisper-language", default=None, help="Force whisper language (e.g. ru)")
    parser.add_argument("--whisper-batch-size", type=int, default=8, help="Batch size for faster-whisper")
    parser.add_argument("--whisper-num-workers", type=int, default=2, help="Num workers for faster-whisper")
    parser.add_argument("--whisper-compute-type", default="float16", help="Compute type for faster-whisper")
    parser.add_argument("--whisper-beam-size", type=int, default=5, help="Beam size for whisper decoding")
    parser.add_argument("--whisper-vad-filter", action="store_true", help="Enable VAD filter for faster-whisper")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N rows (0 disables).",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["none", "count", "whisper", "all"],
        default="none",
        help="Progress verbosity: count prints periodic counters; whisper prints per-row whisper similarity; all prints both.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_path = dataset / "config.json"
    metadata_path = dataset / "metadata_2col.csv"
    wavs_dir = dataset / "wavs"

    errors = []
    if not config_path.exists():
        errors.append("missing_config")
    if not metadata_path.exists():
        errors.append("missing_metadata")
    if not wavs_dir.exists():
        errors.append("missing_wavs_dir")

    config = load_config(config_path)
    expected_sr = None
    if config:
        expected_sr = config.get("audio", {}).get("sample_rate")

    metadata_rows = read_metadata(metadata_path) if metadata_path.exists() else []
    total_rows = len(metadata_rows)

    wav_files = set(p.name for p in wavs_dir.glob("*.wav")) if wavs_dir.exists() else set()

    seen = set()
    missing_wav = []
    duplicate_wav = []
    text_issues = []
    audio_issues = []
    whisper_issues = []
    similarity_values = []

    start_time = time.time()
    whisper_backend = None
    whisper_model = None
    if args.whisper:
        whisper_backend = _select_whisper_backend(args.whisper_backend)
        whisper_model = _load_whisper_model(
            whisper_backend,
            args.whisper_model,
            args.whisper_device,
            args.whisper_compute_type,
        )

    for idx, (row_num, wav_name, text, row_issue) in enumerate(metadata_rows, start=1):
        if args.progress_mode in {"count", "all"} and args.progress_every and idx % args.progress_every == 0:
            elapsed = int(time.time() - start_time)
            rate = (idx / elapsed) if elapsed > 0 else 0.0
            print(
                f"[progress] {idx}/{total_rows} rows, {rate:.2f} rows/s, elapsed={elapsed}s",
                flush=True,
            )
        if row_issue:
            text_issues.append((row_num, wav_name, text, row_issue))
            continue

        if wav_name in seen:
            duplicate_wav.append((row_num, wav_name, text, "duplicate_wav"))
        seen.add(wav_name)

        wav_path = wavs_dir / wav_name
        if not wav_path.exists():
            missing_wav.append((row_num, wav_name, text, "missing_wav"))
            continue

        # Text checks
        if NON_PRINTABLE_RE.search(text):
            text_issues.append((row_num, wav_name, text, "non_printable"))
        text_len = len(text)
        if text_len < args.min_text_len:
            text_issues.append((row_num, wav_name, text, "too_short_text"))
        if text_len > args.max_text_len:
            text_issues.append((row_num, wav_name, text, "too_long_text"))

        letters = len(CYRILLIC_RE.findall(text)) + len(LATIN_RE.findall(text))
        cyr = len(CYRILLIC_RE.findall(text))
        ratio = (cyr / letters) if letters else 0.0
        if ratio < args.min_cyrillic_ratio:
            text_issues.append((row_num, wav_name, text, "low_cyrillic_ratio"))

        # Audio checks
        audio_info = check_audio_info(wav_path, expected_sr)
        if "error" in audio_info:
            audio_issues.append((row_num, wav_name, text, audio_info["error"]))
        else:
            if audio_info.get("issues"):
                for issue in audio_info["issues"]:
                    audio_issues.append((row_num, wav_name, text, issue))

        # Whisper alignment
        if args.whisper:
            try:
                hyp = whisper_transcribe(
                    wav_path,
                    backend=whisper_backend,
                    model=whisper_model,
                    device=args.whisper_device,
                    language=args.whisper_language,
                    batch_size=args.whisper_batch_size,
                    num_workers=args.whisper_num_workers,
                    beam_size=args.whisper_beam_size,
                    vad_filter=args.whisper_vad_filter,
                )
                sim = compute_similarity(normalize_text(text), normalize_text(hyp))
                similarity_values.append(sim)
                if sim < args.similarity_threshold:
                    whisper_issues.append((row_num, wav_name, text, f"low_similarity:{sim:.3f}"))
                if args.progress_mode in {"whisper", "all"}:
                    print(
                        f"[whisper] {idx}/{total_rows} {wav_name} sim={sim:.3f}",
                        flush=True,
                    )
            except Exception as exc:
                whisper_issues.append((row_num, wav_name, text, f"whisper_error:{exc}"))
                if args.progress_mode in {"whisper", "all"}:
                    print(
                        f"[whisper] {idx}/{total_rows} {wav_name} error={exc}",
                        flush=True,
                    )

    extra_wav = sorted(wav_files - seen)

    # Summary
    total_wavs = len(wav_files)
    missing_count = len(missing_wav)
    duplicate_count = len(duplicate_wav)
    extra_count = len(extra_wav)
    text_issue_count = len(text_issues)
    audio_issue_count = len(audio_issues)
    whisper_issue_count = len(whisper_issues)

    whisper_ran = args.whisper
    if args.require_whisper and not whisper_ran:
        errors.append("whisper_not_run")

    min_sim = min(similarity_values) if similarity_values else None
    avg_sim = sum(similarity_values) / len(similarity_values) if similarity_values else None

    failed = (
        errors or missing_count or duplicate_count or extra_count or text_issue_count
        or audio_issue_count or (args.require_whisper and (whisper_issue_count or not whisper_ran))
    )

    verdict = "FAIL" if failed else "PASS"

    # Write TSV
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tsv_path = report_dir / f"validation_{ts}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("row\twav\ttext\tissue\n")
        for row in missing_wav + duplicate_wav + text_issues + audio_issues + whisper_issues:
            f.write("\t".join([str(row[0]), row[1], row[2], row[3]]) + "\n")
        for wav_name in extra_wav:
            f.write("\t".join(["-", wav_name, "", "extra_wav"]) + "\n")

    # Write Markdown report
    md_path = report_dir / f"validation_{ts}.md"
    md = []
    md.append("# Dataset Validation Report\n")
    md.append(f"**Verdict:** {verdict}\n")
    md.append(f"**Dataset:** {dataset}\n")
    md.append(f"**Date (UTC):** {datetime.utcnow().isoformat()}\n")
    md.append("\n## Summary\n")
    md.append(f"- Total WAV: {total_wavs}\n")
    md.append(f"- Total metadata rows: {total_rows}\n")
    md.append(f"- Missing WAV: {missing_count}\n")
    md.append(f"- Extra WAV: {extra_count}\n")
    md.append(f"- Duplicates: {duplicate_count}\n")
    md.append(f"- Text issues: {text_issue_count}\n")
    md.append(f"- Audio issues: {audio_issue_count}\n")
    if whisper_ran:
        md.append(f"- Whisper issues: {whisper_issue_count}\n")
        if min_sim is not None:
            md.append(f"- Min similarity: {min_sim:.3f}\n")
        if avg_sim is not None:
            md.append(f"- Avg similarity: {avg_sim:.3f}\n")
    else:
        md.append("- Whisper: not run\n")

    if errors:
        md.append("\n## Errors\n")
        for err in errors:
            md.append(f"- {err}\n")

    md.append("\n## Artifacts\n")
    md.append(f"- TSV: {tsv_path.name}\n")

    md_path.write_text("".join(md), encoding="utf-8")

    print(f"Report: {md_path}")
    print(f"Issues TSV: {tsv_path}")
    print(f"Verdict: {verdict}")

    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
