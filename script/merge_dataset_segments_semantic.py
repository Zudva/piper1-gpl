#!/usr/bin/env python3
"""Merge too-short segmented clips into more meaningful phrases.

This is intended for datasets produced by Whisper timestamp segmentation where you
sometimes get 1s / one-word fragments (e.g. "мире."). Those fragments are not
playback artifacts; they are real clips. To reduce them, we merge consecutive
segments *within the same source prefix* (e.g. felix_000399_0026 + _0027 + ...)
into longer, more semantically complete phrases.

Heuristic (semantic-ish, local-only):
- Prefer to end a merged clip on sentence punctuation (., !, ?, …)
- Keep merging until we reach a soft target duration OR a good boundary
- Never exceed a hard max duration
- If a chunk would be very short, force-merge with neighbors

Output dataset layout:
  <out_dataset>/
    config.json
    metadata_2col.csv
    wavs/
      <prefix>_m0000.wav
      ...

Local-only. Requires ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


_BOUNDARY_RE_DEFAULT = r"[.!?…]+[\"\')\]]*$"  # sentence-ending punctuation, optionally followed by quotes/brackets
_NAME_RE = re.compile(r"^(?P<prefix>.+?)_(?P<seg>\d+)\.wav$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Item:
    wav: str
    text: str
    wav_path: Path
    prefix: str
    seg: int
    dur_s: float


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", " ")
    return _WS_RE.sub(" ", text).strip()


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate() or 1
        return wf.getnframes() / float(rate)


def _concat_wav_python(*, inputs: list[Path], out_wav: Path) -> bool:
    """Concatenate WAVs quickly in Python.

    Returns True if successful. Returns False if input WAV formats are not
    identical (channels/sample_rate/sample_width/comp_type), in which case a
    re-encode via ffmpeg is safer.
    """

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    params = None
    frames: list[bytes] = []
    for p in inputs:
        with wave.open(str(p), "rb") as wf:
            p_params = wf.getparams()
            if params is None:
                params = p_params
            else:
                if (
                    p_params.nchannels != params.nchannels
                    or p_params.sampwidth != params.sampwidth
                    or p_params.framerate != params.framerate
                    or p_params.comptype != params.comptype
                    or p_params.compname != params.compname
                ):
                    return False
            frames.append(wf.readframes(wf.getnframes()))

    if params is None:
        return False

    with wave.open(str(out_wav), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(b"".join(frames))

    return True


def _load_config(dataset_dir: Path) -> dict:
    cfg_path = dataset_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _iter_metadata(dataset_dir: Path) -> Iterable[tuple[str, str]]:
    meta = dataset_dir / "metadata_2col.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"Missing metadata_2col.csv: {meta}")
    with meta.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row or len(row) < 2:
                continue
            wav = row[0].strip()
            text = _normalize_text(row[1])
            if not wav or not text:
                continue
            yield wav, text


def _parse_name(wav: str) -> tuple[str, int]:
    m = _NAME_RE.match(wav)
    if not m:
        raise ValueError(f"Unexpected wav name (expected <prefix>_<seg>.wav): {wav}")
    return m.group("prefix"), int(m.group("seg"))


def _is_boundary(text: str, boundary_re: re.Pattern[str]) -> bool:
    return bool(boundary_re.search(text.strip()))


def _ffmpeg_concat_to_wav(
    *,
    inputs: list[Path],
    out_wav: Path,
    sample_rate: int,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH")

    # Use concat demuxer via temp list file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        list_path = Path(tf.name)
        for p in inputs:
            tf.write(f"file '{p.as_posix()}'\n")

    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(out_wav),
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path, help="Prepared dataset dir (config.json, wavs/, metadata_2col.csv)")
    p.add_argument(
        "--out-dataset",
        type=Path,
        default=None,
        help="Output dataset dir (default: <dataset>_merged_<ts>)",
    )

    # Soft targets
    p.add_argument("--target-min-seconds", type=float, default=2.5, help="Try to reach at least this duration before cutting")
    p.add_argument("--target-max-seconds", type=float, default=8.0, help="Prefer to cut around/under this duration when possible")

    # Hard constraints
    p.add_argument("--max-seg-seconds", type=float, default=15.0, help="Hard cap for any merged clip")

    # Force-merge small fragments
    p.add_argument("--force-merge-below-seconds", type=float, default=1.6, help="Always merge if current chunk is shorter than this")
    p.add_argument("--force-merge-below-chars", type=int, default=10, help="Always merge if current chunk has fewer than this many chars")

    p.add_argument("--boundary-regex", type=str, default=_BOUNDARY_RE_DEFAULT, help="Regex for semantic boundary (default: sentence punctuation)")

    p.add_argument(
        "--concat-backend",
        choices=["python", "ffmpeg"],
        default="python",
        help="How to concatenate audio (python is much faster if WAV formats match)",
    )

    p.add_argument("--limit-prefixes", type=int, default=0, help="If >0, only process first N prefixes (debug)")
    p.add_argument("--dry-run", action="store_true", help="Do not write audio; only print stats")

    args = p.parse_args()

    dataset = args.dataset.expanduser().resolve()
    wavs_dir = dataset / "wavs"
    if not wavs_dir.is_dir():
        raise SystemExit(f"Missing wavs/: {wavs_dir}")

    cfg = _load_config(dataset)
    sample_rate = int(cfg.get("audio", {}).get("sample_rate") or 22050)

    boundary_re = re.compile(args.boundary_regex)

    # Load items
    items: list[Item] = []
    for wav, text in _iter_metadata(dataset):
        wav_path = wavs_dir / wav
        if not wav_path.is_file():
            continue
        try:
            prefix, seg = _parse_name(wav)
        except Exception:
            # Skip unexpected names
            continue
        try:
            dur = _wav_duration_seconds(wav_path)
        except Exception:
            continue
        items.append(Item(wav=wav, text=text, wav_path=wav_path, prefix=prefix, seg=seg, dur_s=dur))

    # Group by prefix
    items.sort(key=lambda x: (x.prefix, x.seg))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dataset = (
        args.out_dataset.expanduser().resolve()
        if args.out_dataset
        else Path(f"{dataset}_merged_{ts}").resolve()
    )

    if not args.dry_run:
        out_dataset.mkdir(parents=True, exist_ok=False)
        (out_dataset / "wavs").mkdir(parents=True, exist_ok=True)
        (out_dataset / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_rows: list[tuple[str, str]] = []

    prefixes_seen = 0
    i = 0
    merged_count = 0
    dropped = 0
    total_in = len(items)

    while i < total_in:
        prefix = items[i].prefix
        group: list[Item] = []
        while i < total_in and items[i].prefix == prefix:
            group.append(items[i])
            i += 1

        prefixes_seen += 1
        if args.limit_prefixes and prefixes_seen > args.limit_prefixes:
            break

        # Merge within group
        g_idx = 0
        chunks: list[tuple[list[Item], str]] = []

        while g_idx < len(group):
            chunk_items: list[Item] = []
            chunk_text_parts: list[str] = []
            chunk_dur = 0.0

            while g_idx < len(group):
                it = group[g_idx]
                next_dur = chunk_dur + it.dur_s
                next_text = " ".join(chunk_text_parts + [it.text]).strip()

                # If adding this item would exceed hard max, stop before it (unless chunk is empty)
                if chunk_items and next_dur > args.max_seg_seconds:
                    break

                chunk_items.append(it)
                chunk_text_parts.append(it.text)
                chunk_dur = next_dur
                g_idx += 1

                # Decide whether to cut here.
                end_is_boundary = _is_boundary(it.text, boundary_re)

                too_small = (chunk_dur < args.force_merge_below_seconds) or (len(next_text) < args.force_merge_below_chars)
                reached_target = chunk_dur >= args.target_min_seconds
                prefer_cut = chunk_dur >= args.target_max_seconds

                # If very small, keep merging no matter what.
                if too_small:
                    continue

                # If we reached the min target and have a semantic boundary, cut.
                if reached_target and end_is_boundary:
                    break

                # If we're getting long, cut at the first reasonable boundary.
                if prefer_cut and end_is_boundary:
                    break

            merged_text = _normalize_text(" ".join(chunk_text_parts))
            if not merged_text or not chunk_items:
                dropped += 1
                continue
            chunks.append((chunk_items, merged_text))

        # Tail fixup: absorb a very short last chunk into the previous one.
        if len(chunks) >= 2:
            last_items, last_text = chunks[-1]
            last_dur = sum(x.dur_s for x in last_items)
            if (last_dur < args.force_merge_below_seconds) or (len(last_text) < args.force_merge_below_chars):
                prev_items, prev_text = chunks[-2]
                chunks[-2] = (prev_items + last_items, _normalize_text(prev_text + " " + last_text))
                chunks.pop()

        m_idx = 0
        for chunk_items, merged_text in chunks:
            out_name = f"{prefix}_m{m_idx:04d}.wav"
            m_idx += 1

            if not args.dry_run:
                out_wav = out_dataset / "wavs" / out_name
                in_paths = [x.wav_path for x in chunk_items]
                if args.concat_backend == "python":
                    ok = _concat_wav_python(inputs=in_paths, out_wav=out_wav)
                    if not ok:
                        _ffmpeg_concat_to_wav(inputs=in_paths, out_wav=out_wav, sample_rate=sample_rate)
                else:
                    _ffmpeg_concat_to_wav(inputs=in_paths, out_wav=out_wav, sample_rate=sample_rate)

            out_rows.append((out_name, merged_text))
            merged_count += 1

    if args.dry_run:
        print(f"DRY RUN")
        print(f"Input dataset: {dataset}")
        print(f"Input items:   {total_in}")
        print(f"Merged clips:  {merged_count}")
        print(f"Dropped:       {dropped}")
        return 0

    meta_out = out_dataset / "metadata_2col.csv"
    with meta_out.open("w", encoding="utf-8") as f:
        for wav, text in out_rows:
            f.write(f"{wav}|{text}\n")

    info = out_dataset / "MERGE_INFO.txt"
    info.write_text(
        "\n".join(
            [
                f"source_dataset={dataset}",
                f"sample_rate={sample_rate}",
                f"target_min_seconds={args.target_min_seconds}",
                f"target_max_seconds={args.target_max_seconds}",
                f"max_seg_seconds={args.max_seg_seconds}",
                f"force_merge_below_seconds={args.force_merge_below_seconds}",
                f"force_merge_below_chars={args.force_merge_below_chars}",
                f"boundary_regex={args.boundary_regex}",
                f"merged_clips={merged_count}",
                f"dropped={dropped}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote merged dataset: {out_dataset}")
    print(f"Clips: {len(out_rows)}")
    print(f"Metadata: {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
