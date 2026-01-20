#!/usr/bin/env python3
"""Stage B: Align Stage A chunks to audio and emit a cutlist (WhisperX).

Inputs:
- Stage A JSON produced by script/text_splitter/01_text_splitter.py

Output:
- JSONL cutlist with records:
    {"src_audio": "wavs/....wav", "start": 12.345, "end": 15.678, "text": "...", "sim": 0.93}

Notes:
- Local-only. No paid APIs.
- This script can be compute-heavy when executed. Use --limit/--dry-run for spot checks.
- Matching is heuristic: WhisperX provides word timestamps for ASR transcript; we then try
  to find each target chunk inside the ASR words in order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


@dataclass(frozen=True)
class Word:
    token: str
    start: float
    end: float
    raw: str


def _normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", " ")
    text = text.lower()
    # Optional: unify ё/е to reduce mismatch
    text = text.replace("ё", "е")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    text = _normalize_text(text)
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def _iter_stage_a_items(path: Path) -> Iterable[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Stage A JSON must be a list: {path}")
    for item in data:
        if isinstance(item, dict):
            yield item


def _load_words_whisperx(
    *,
    audio_path: Path,
    device: str,
    whisper_model: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> list[Word]:
    try:
        import whisperx  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency 'whisperx'. Install it in your environment before running Stage B."
        )

    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(
        whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # WhisperX align model is language-specific. If language not given, prefer detected.
    lang_code = language or result.get("language")
    if not lang_code:
        raise SystemExit("WhisperX did not return a language; pass --language")

    align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    aligned = whisperx.align(
        result.get("segments", []),
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    out: list[Word] = []
    for seg in aligned.get("segments", []) if isinstance(aligned, dict) else []:
        words = seg.get("words") if isinstance(seg, dict) else None
        if not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, dict):
                continue
            raw = str(w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not raw:
                continue
            if start is None or end is None:
                continue
            try:
                start_f = float(start)
                end_f = float(end)
            except Exception:
                continue
            if end_f <= start_f:
                continue
            toks = _tokenize(raw)
            if not toks:
                continue
            # WhisperX 'word' sometimes includes punctuation; keep first token as anchor
            out.append(Word(token=toks[0], start=start_f, end=end_f, raw=raw))

    return out


@dataclass(frozen=True)
class Match:
    start_i: int
    end_i: int  # exclusive
    sim: float


def _best_match_sequential(
    *,
    transcript: list[Word],
    target_tokens: list[str],
    start_from: int,
    max_extra_tokens: int,
    max_candidates: int,
) -> Match | None:
    if not target_tokens:
        return None

    # Index positions by token to reduce brute force.
    positions: list[int] = []
    first = target_tokens[0]
    for i in range(start_from, len(transcript)):
        if transcript[i].token == first:
            positions.append(i)
            if max_candidates and len(positions) >= max_candidates:
                break

    # If the first token never appears, fall back to a sparse scan.
    if not positions:
        step = max(1, int(len(transcript) / 2000))  # ~2k probes max
        positions = list(range(start_from, len(transcript), step))

    tgt_len = len(target_tokens)
    min_len = max(1, tgt_len - max_extra_tokens)
    max_len = tgt_len + max_extra_tokens

    tgt_join = " ".join(target_tokens)
    best: Match | None = None

    for start_i in positions:
        for L in range(min_len, max_len + 1):
            end_i = start_i + L
            if end_i > len(transcript):
                break
            cand_tokens = [w.token for w in transcript[start_i:end_i]]
            cand_join = " ".join(cand_tokens)
            sim = _similarity(cand_join, tgt_join)
            if best is None or sim > best.sim:
                best = Match(start_i=start_i, end_i=end_i, sim=sim)

    return best


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--to-align", required=True, type=Path, help="Stage A JSON (to_align.json)")
    p.add_argument(
        "--audio-root",
        required=True,
        type=Path,
        help="Directory that contains the Stage A audio_path (e.g. .../datasets/elevenlabs/<voice_id>/)",
    )
    p.add_argument("--out", required=True, type=Path, help="Output cutlist.jsonl")

    p.add_argument("--device", default="cuda", help="cpu|cuda")
    p.add_argument("--whisper-model", default="large-v3", help="WhisperX ASR model name")
    p.add_argument(
        "--compute-type",
        default="float16",
        help="WhisperX compute type (e.g. float16, int8, int8_float16)",
    )
    p.add_argument("--language", default=None, help="Language code (e.g. ru). If omitted, use WhisperX detection")
    p.add_argument("--batch-size", type=int, default=8, help="Transcription batch size (VRAM-sensitive)")

    p.add_argument(
        "--min-sim",
        type=float,
        default=0.80,
        help="Minimum similarity to accept a match (default: 0.80)",
    )
    p.add_argument(
        "--max-extra-tokens",
        type=int,
        default=4,
        help="Allow window length +/- this many tokens when matching (default: 4)",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=2000,
        help="Max candidate start positions per chunk (default: 2000)",
    )

    p.add_argument(
        "--pad-seconds",
        type=float,
        default=0.10,
        help="Pad start/end by this many seconds (default: 0.10)",
    )

    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N Stage A items")
    p.add_argument("--dry-run", action="store_true", help="Do not write output; print a small summary")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")

    args = p.parse_args()

    to_align = args.to_align.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    if not to_align.is_file():
        raise SystemExit(f"Missing --to-align: {to_align}")
    if not audio_root.is_dir():
        raise SystemExit(f"Missing --audio-root dir: {audio_root}")

    if out_path.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists (use --overwrite): {out_path}")

    items = list(_iter_stage_a_items(to_align))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    total_chunks = 0
    matched_chunks = 0
    unresolved_chunks = 0

    out_lines: list[str] = []

    for item_i, item in enumerate(items, start=1):
        audio_rel = item.get("audio_path")
        chunks = item.get("sentences")
        if not isinstance(audio_rel, str) or not audio_rel.strip():
            continue
        if not isinstance(chunks, list) or not chunks:
            continue

        audio_path = (audio_root / audio_rel).resolve()
        if not audio_path.is_file():
            raise SystemExit(f"Missing audio file: {audio_path} (from audio_path={audio_rel})")

        words = _load_words_whisperx(
            audio_path=audio_path,
            device=args.device,
            whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            language=args.language,
            batch_size=int(args.batch_size),
        )
        if not words:
            raise SystemExit(f"No aligned words produced for: {audio_path}")

        cursor = 0

        for chunk in chunks:
            if not isinstance(chunk, str):
                continue
            chunk = chunk.strip()
            if not chunk:
                continue

            total_chunks += 1
            tgt_tokens = _tokenize(chunk)
            m = _best_match_sequential(
                transcript=words,
                target_tokens=tgt_tokens,
                start_from=cursor,
                max_extra_tokens=int(args.max_extra_tokens),
                max_candidates=int(args.max_candidates),
            )

            if m is None or m.sim < float(args.min_sim):
                unresolved_chunks += 1
                rec = {
                    "src_audio": audio_rel,
                    "start": None,
                    "end": None,
                    "text": chunk,
                    "sim": float(m.sim) if m else None,
                    "status": "unmatched",
                }
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                continue

            start_t = words[m.start_i].start - float(args.pad_seconds)
            end_t = words[m.end_i - 1].end + float(args.pad_seconds)
            start_t = _clamp(start_t, 0.0, 1e9)
            end_t = _clamp(end_t, 0.0, 1e9)
            if end_t <= start_t:
                unresolved_chunks += 1
                rec = {
                    "src_audio": audio_rel,
                    "start": None,
                    "end": None,
                    "text": chunk,
                    "sim": float(m.sim),
                    "status": "bad_times",
                }
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                continue

            matched_chunks += 1
            cursor = max(cursor, m.end_i)

            rec = {
                "src_audio": audio_rel,
                "start": round(float(start_t), 3),
                "end": round(float(end_t), 3),
                "text": chunk,
                "sim": round(float(m.sim), 4),
                "status": "ok",
            }
            out_lines.append(json.dumps(rec, ensure_ascii=False))

        if item_i % 1 == 0:
            print(
                f"[{item_i}/{len(items)}] audio={audio_rel} chunks_total={total_chunks} ok={matched_chunks} unmatched={unresolved_chunks}",
                flush=True,
            )

    if args.dry_run:
        print("DRY RUN: not writing cutlist")
        print(f"Stage A items: {len(items)}")
        print(f"Chunks: total={total_chunks} ok={matched_chunks} unresolved={unresolved_chunks}")
        if out_lines:
            print("Sample output lines:")
            for line in out_lines[: min(5, len(out_lines))]:
                print(line)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    print(f"Wrote cutlist: {out_path}")
    print(f"Chunks: total={total_chunks} ok={matched_chunks} unresolved={unresolved_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
