#!/usr/bin/env python3
"""Filter a Piper-style dataset by simple rules (duration/text) without re-segmentation.

This is useful when a prepared dataset contains unwanted short clips (e.g., ~1.0s)
caused by Whisper segmentation settings.

Output dataset layout:
  <out_dataset>/
    config.json
    metadata_2col.csv
    wavs -> symlink to <input>/wavs

Local-only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import wave
from datetime import datetime
from pathlib import Path


_WS_RE = re.compile(r"\s+")


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate() or 1
        return wf.getnframes() / float(rate)


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path, help="Input dataset root")
    p.add_argument(
        "--out-dataset",
        type=Path,
        default=None,
        help="Output dataset dir (default: <dataset>_filtered_rules_<ts>)",
    )
    p.add_argument("--min-duration", type=float, default=2.0, help="Drop clips shorter than this many seconds")
    p.add_argument("--max-duration", type=float, default=15.0, help="Drop clips longer than this many seconds")
    p.add_argument("--min-text-chars", type=int, default=6, help="Drop clips with shorter text")
    p.add_argument("--max-text-chars", type=int, default=400, help="Drop clips with longer text")
    p.add_argument(
        "--drop-regex",
        type=str,
        default=r"^[\W_]+$",
        help="Drop if normalized text matches this regex (default: punctuation-only)",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, only scan first N rows (debug)")

    args = p.parse_args()

    dataset = args.dataset.expanduser().resolve()
    meta_in = dataset / "metadata_2col.csv"
    cfg_in = dataset / "config.json"
    wavs_in = dataset / "wavs"

    if not meta_in.is_file():
        raise SystemExit(f"Missing metadata_2col.csv: {meta_in}")
    if not cfg_in.is_file():
        raise SystemExit(f"Missing config.json: {cfg_in}")
    if not wavs_in.is_dir():
        raise SystemExit(f"Missing wavs/: {wavs_in}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dataset = (
        args.out_dataset.expanduser().resolve()
        if args.out_dataset
        else Path(f"{dataset}_filtered_rules_{ts}").resolve()
    )

    out_dataset.mkdir(parents=True, exist_ok=False)
    (out_dataset / "config.json").write_text(cfg_in.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dataset / "wavs").symlink_to(wavs_in)

    drop_re = re.compile(args.drop_regex) if args.drop_regex else None

    kept = 0
    dropped = 0
    reasons: dict[str, int] = {}

    out_meta = out_dataset / "metadata_2col.csv"
    with meta_in.open("r", encoding="utf-8") as f_in, out_meta.open("w", encoding="utf-8") as f_out:
        reader = csv.reader(f_in, delimiter="|")
        for i, row in enumerate(reader, start=1):
            if args.limit and i > args.limit:
                break
            if not row or len(row) < 2:
                dropped += 1
                reasons["bad_row"] = reasons.get("bad_row", 0) + 1
                continue
            wav = row[0].strip()
            text = _normalize_text(row[1])
            if not wav:
                dropped += 1
                reasons["empty_wav"] = reasons.get("empty_wav", 0) + 1
                continue

            wav_path = wavs_in / wav
            if not wav_path.is_file():
                dropped += 1
                reasons["missing_wav"] = reasons.get("missing_wav", 0) + 1
                continue

            try:
                dur = _wav_duration_seconds(wav_path)
            except Exception:
                dropped += 1
                reasons["invalid_wav"] = reasons.get("invalid_wav", 0) + 1
                continue

            if dur < args.min_duration:
                dropped += 1
                reasons["too_short"] = reasons.get("too_short", 0) + 1
                continue
            if dur > args.max_duration:
                dropped += 1
                reasons["too_long"] = reasons.get("too_long", 0) + 1
                continue

            if len(text) < args.min_text_chars:
                dropped += 1
                reasons["text_too_short"] = reasons.get("text_too_short", 0) + 1
                continue
            if len(text) > args.max_text_chars:
                dropped += 1
                reasons["text_too_long"] = reasons.get("text_too_long", 0) + 1
                continue

            if drop_re and drop_re.match(text):
                dropped += 1
                reasons["text_regex"] = reasons.get("text_regex", 0) + 1
                continue

            f_out.write(f"{wav}|{text}\n")
            kept += 1

    print(f"Input dataset:  {dataset}")
    print(f"Output dataset: {out_dataset}")
    print(f"Rules: min_dur={args.min_duration}s max_dur={args.max_duration}s min_chars={args.min_text_chars}")
    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")
    if reasons:
        print("Drop reasons:")
        for k in sorted(reasons):
            print(f"  {k}: {reasons[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
