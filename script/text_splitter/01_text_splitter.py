#!/usr/bin/env python3
"""Smart text splitting for long monologues (Stage A: text only).

Reads an ElevenLabs-style manifest.jsonl (audio_path + text) and produces a JSON
file suitable for the next Alignment stage (WhisperX/MFA):

[
  {
    "audio_path": "wavs/example.wav",
    "sentences": ["...", "..."],
    "original_full_text": "...",
    "manifest_meta": {...}
  }
]

This script focuses on semantic-ish text chunking for Russian using `razdel`
(sentenize). It does NOT cut audio.

Local-only. No paid APIs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_WS_RE = re.compile(r"\s+")


def _split_long_sentence(text: str, max_len: int) -> list[str]:
    """Split an overlong sentence into smaller parts.

    Strategy: prefer splitting on strong intra-sentence separators near the
    target length, falling back to whitespace.
    """

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    remaining = text

    # Prefer these break characters (ordered)
    break_chars = [";", ":", ",", "—"]

    while remaining and len(remaining) > max_len:
        window = remaining[: max_len + 1]

        cut = -1
        # Try to find a separator between 70%..100% of max_len
        start_search = int(max_len * 0.70)
        for ch in break_chars:
            idx = window.rfind(ch, start_search)
            if idx != -1:
                cut = idx + 1  # keep punctuation
                break

        # Fallback: cut at last whitespace
        if cut == -1:
            idx = window.rfind(" ", start_search)
            if idx != -1:
                cut = idx
            else:
                cut = max_len

        head = remaining[:cut].strip()
        if head:
            parts.append(head)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def clean_text(text: str) -> str:
    """Basic normalization for text coming from manifests."""
    # Collapse all whitespace/newlines
    text = _WS_RE.sub(" ", text)
    # Remove spaces before punctuation (common parsing/OCR artifact)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Normalize hyphen-as-dash: " - " -> " — "
    text = re.sub(r"\s-\s", " — ", text)
    # Remove separator used in Piper metadata
    text = text.replace("|", " ")
    return text.strip()


def smart_grouping(*, sentences: list[str], max_len: int, min_len: int) -> list[str]:
    """Group sentence strings into chunks to fit length constraints."""
    # First, expand any single overlong sentences.
    expanded: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        expanded.extend(_split_long_sentence(s, max_len))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent_text in expanded:
        sent_text = sent_text.strip()
        if not sent_text:
            continue

        sent_len = len(sent_text)

        # If the next sentence fits, append
        if current and (current_len + 1 + sent_len) <= max_len:
            current.append(sent_text)
            current_len += 1 + sent_len
            continue

        if not current and sent_len <= max_len:
            current = [sent_text]
            current_len = sent_len
            continue

        # If we are here: would exceed max_len OR sentence itself is too long.
        # Close current chunk if present.
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

        # Start new chunk with this sentence.
        current = [sent_text]
        current_len = sent_len

    # Tail
    if current:
        last = " ".join(current).strip()
        if len(last) < min_len and chunks:
            prev = chunks.pop()
            # allow slight overflow to avoid "orphans"
            if len(prev) + 1 + len(last) <= (max_len + 50):
                chunks.append((prev + " " + last).strip())
            else:
                chunks.append(prev)
                chunks.append(last)
        else:
            chunks.append(last)

    # Final cleanup (ensure no empties)
    chunks = [c for c in (c.strip() for c in chunks) if c]
    return chunks


def _load_manifest_line(line: str) -> dict[str, Any] | None:
    line = line.strip("\n")
    if not line.strip():
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def process_manifest(
    *,
    input_file: Path,
    output_file: Path,
    target_len: int,
    min_len: int,
    limit: int = 0,
) -> int:
    try:
        from razdel import sentenize  # type: ignore
    except Exception:
        print("ERROR: missing dependency 'razdel'. Install: pip install razdel", file=sys.stderr)
        return 2

    input_file = input_file.expanduser().resolve()
    if not input_file.is_file():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        return 2

    output: list[dict[str, Any]] = []
    processed = 0
    kept = 0

    seen_nonempty = 0

    with input_file.open("r", encoding="utf-8") as f_in:
        for idx, line in enumerate(f_in, start=1):
            if line.strip():
                seen_nonempty += 1
                if limit and seen_nonempty > limit:
                    break
            record = _load_manifest_line(line)
            if not record:
                continue

            processed += 1

            original_text = record.get("text")
            audio_path = record.get("audio_path")
            if not isinstance(original_text, str) or not original_text.strip():
                continue
            if not isinstance(audio_path, str) or not audio_path.strip():
                continue

            cleaned = clean_text(original_text)
            if not cleaned:
                continue

            # razdel sentence segmentation
            sents = [s.text for s in sentenize(cleaned)]
            segments = smart_grouping(sentences=sents, max_len=target_len, min_len=min_len)
            if not segments:
                continue

            meta_keep = {
                "line": idx,
                "lang": record.get("lang"),
                "voice_id": record.get("voice_id"),
                "source": record.get("source"),
                "sample_rate": record.get("sample_rate"),
                "time_unix": record.get("time_unix"),
                "hash": record.get("hash"),
                "session": record.get("session"),
                "meta": record.get("meta"),
            }

            output.append(
                {
                    "audio_path": audio_path,
                    "sentences": segments,
                    "original_full_text": cleaned,
                    "manifest_meta": meta_keep,
                }
            )
            kept += 1

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OK: processed_lines={processed} kept_records={kept}")
    print(f"Wrote: {output_file}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Path to manifest.jsonl")
    p.add_argument("--output", required=True, type=Path, help="Path to output JSON (e.g. to_align.json)")
    p.add_argument("--target-length", type=int, default=160, help="Target max chunk length in characters")
    p.add_argument("--min-length", type=int, default=35, help="Minimum chunk length to avoid orphans")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N manifest lines (debug)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file")
    p.add_argument("--print-samples", type=int, default=0, help="Print first N output entries (debug)")

    args = p.parse_args()

    if args.target_length < 40:
        print("ERROR: --target-length is too small", file=sys.stderr)
        return 2
    if args.min_length < 10:
        print("ERROR: --min-length is too small", file=sys.stderr)
        return 2

    out_path = args.output.expanduser().resolve()
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: output exists (use --overwrite): {out_path}", file=sys.stderr)
        return 2

    limit = int(args.limit) if args.limit else 0
    print_samples = int(args.print_samples) if args.print_samples else 0

    # Run main processing
    rc = process_manifest(
        input_file=args.input,
        output_file=out_path,
        target_len=args.target_length,
        min_len=args.min_length,
        limit=limit,
    )
    if rc != 0:
        return rc

    if limit or print_samples:
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return 0
            if print_samples:
                for i, entry in enumerate(data[:print_samples]):
                    ap = entry.get("audio_path")
                    sents = entry.get("sentences")
                    print(f"\n--- sample {i+1} ---")
                    print(f"audio_path: {ap}")
                    if isinstance(sents, list):
                        for j, s in enumerate(sents[:10]):
                            print(f"  {j+1:02d}. {s}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
