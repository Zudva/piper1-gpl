#!/usr/bin/env python3
"""Build a training-ready Piper dataset from a cutlist JSONL.

Input:
  - cutlist.jsonl with rows like {src_audio, start, end, text, review?}
  - audio_root that contains (or is) the root for src_audio relative paths

Output dataset layout:
  <out_dataset>/
    wavs/*.wav           (mono, PCM_16, resampled)
    metadata_2col.csv    (wav|text) where wav is a filename under wavs/
    config.json          (at least audio.sample_rate)

Notes:
  - This script never modifies the source audio.
  - If row.review.replaced_audio.new_audio_path exists, that file is used as the source
    and the row is treated as already-segmented (start/end ignored).

Local-only. Uses ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CutItem:
    line_num: int
    src_audio: str
    start: float | None
    end: float | None
    text: str
    verdict: str | None
    replaced_audio_path: str | None


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _iter_cutlist(path: Path) -> Iterable[CutItem]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            text = str(row.get("text") or "").strip()
            src_audio = str(row.get("src_audio") or "").strip()
            start = _to_float(row.get("start"))
            end = _to_float(row.get("end"))

            verdict = None
            review = row.get("review")
            if isinstance(review, dict):
                verdict = (review.get("verdict") or None)

            replaced_audio_path = None
            if isinstance(review, dict):
                rep = review.get("replaced_audio")
                if isinstance(rep, dict):
                    rap = rep.get("new_audio_path")
                    if rap:
                        replaced_audio_path = str(rap)

            yield CutItem(
                line_num=line_num,
                src_audio=src_audio,
                start=start,
                end=end,
                text=text,
                verdict=str(verdict).strip() if verdict is not None else None,
                replaced_audio_path=replaced_audio_path,
            )


def _load_to_align(path: Path) -> dict[str, list[str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read to_align JSON: {path} ({e})")
    if not isinstance(raw, list):
        raise SystemExit(f"to_align must be a JSON list: {path}")

    out: dict[str, list[str]] = {}
    for it in raw:
        if not isinstance(it, dict):
            continue
        ap = str(it.get("audio_path") or "").strip()
        if not ap:
            continue
        sents_raw = it.get("sentences")
        if not isinstance(sents_raw, list):
            continue
        sents = [str(s).strip() for s in sents_raw if str(s).strip()]
        out[ap] = sents
    return out


def _ffmpeg_cut_resample(
    *,
    src_wav: Path,
    out_wav: Path,
    start: float | None,
    end: float | None,
    sample_rate: int,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_wav),
    ]
    if start is not None:
        cmd += ["-ss", str(float(start))]
    if end is not None:
        cmd += ["-to", str(float(end))]

    cmd += [
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write_config(out_dataset: Path, sample_rate: int) -> None:
    cfg = {"audio": {"sample_rate": int(sample_rate)}}
    (out_dataset / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutlist", required=True, type=Path, help="Path to cutlist.jsonl")
    p.add_argument("--audio-root", required=True, type=Path, help="Root directory for src_audio paths")
    p.add_argument(
        "--to-align",
        type=Path,
        default=None,
        help=(
            "Optional path to to_align.json (Stage A). When provided, text can be sourced from it to avoid drift."
        ),
    )
    p.add_argument(
        "--text-source",
        choices=["auto", "cutlist", "to_align"],
        default="auto",
        help=(
            "Where to take training text from. "
            "auto=use to_align if provided, else cutlist; cutlist=use cutlist.text; to_align=use to_align sentences by order."
        ),
    )
    p.add_argument(
        "--warn-on-text-mismatch",
        action="store_true",
        help="If using to_align, warn when cutlist.text differs from the corresponding to_align sentence.",
    )
    p.add_argument(
        "--out-dataset",
        required=True,
        type=Path,
        help="Output dataset directory (will contain wavs/, metadata_2col.csv, config.json)",
    )
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N kept rows")
    p.add_argument(
        "--drop-verdicts",
        default="drop,bad,skip,missing",
        help="Comma-separated review verdicts to skip (default: drop,bad,skip,missing)",
    )
    p.add_argument(
        "--skip-missing-times",
        action="store_true",
        help="Skip rows without start/end (default: fail the row)",
    )

    args = p.parse_args()

    cutlist = args.cutlist.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    out_dataset = args.out_dataset.expanduser().resolve()
    to_align_path = args.to_align.expanduser().resolve() if args.to_align else None

    if not cutlist.is_file():
        raise SystemExit(f"Cutlist not found: {cutlist}")
    if not audio_root.is_dir():
        raise SystemExit(f"Audio root not found: {audio_root}")

    to_align_map: dict[str, list[str]] | None = None
    if to_align_path:
        if not to_align_path.is_file():
            raise SystemExit(f"to_align not found: {to_align_path}")
        to_align_map = _load_to_align(to_align_path)

    use_to_align_text = args.text_source == "to_align" or (args.text_source == "auto" and to_align_map is not None)
    if args.text_source == "to_align" and to_align_map is None:
        raise SystemExit("--text-source to_align requires --to-align")

    if out_dataset.exists():
        if args.overwrite:
            # Keep it simple and safe: only delete expected files/dirs
            for name in [
                "metadata_2col.csv",
                "config.json",
                "source_cutlist.jsonl",
                "source_to_align.json",
                "segments.jsonl",
            ]:
                pth = out_dataset / name
                if pth.exists():
                    pth.unlink()
            wavs_dir = out_dataset / "wavs"
            if wavs_dir.is_dir():
                for wav in wavs_dir.glob("*.wav"):
                    wav.unlink()
        else:
            raise SystemExit(f"Output dataset exists (use --overwrite): {out_dataset}")

    out_dataset.mkdir(parents=True, exist_ok=True)
    wavs_dir = out_dataset / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    _write_config(out_dataset, args.sample_rate)
    (out_dataset / "source_cutlist.jsonl").write_text(cutlist.read_text(encoding="utf-8"), encoding="utf-8")
    if to_align_path:
        (out_dataset / "source_to_align.json").write_text(
            to_align_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    drop_verdicts = {v.strip().lower() for v in (args.drop_verdicts or "").split(",") if v.strip()}

    meta_path = out_dataset / "metadata_2col.csv"
    kept = 0
    skipped = 0
    failed = 0
    warned = 0

    # For each src_audio, consume to_align sentences by order so cutlist rows and to_align stay in sync.
    to_align_cursor: dict[str, int] = {}

    seg_path = out_dataset / "segments.jsonl"

    with meta_path.open("w", encoding="utf-8") as meta, seg_path.open("w", encoding="utf-8") as seg:
        for item in _iter_cutlist(cutlist):
            to_align_text: str | None = None
            to_align_idx: int | None = None
            if use_to_align_text and to_align_map is not None and item.src_audio:
                sents = to_align_map.get(item.src_audio)
                if sents is not None:
                    idx = to_align_cursor.get(item.src_audio, 0)
                    to_align_cursor[item.src_audio] = idx + 1
                    to_align_idx = idx
                    if idx < len(sents):
                        to_align_text = sents[idx]

            if item.verdict and item.verdict.lower() in drop_verdicts:
                skipped += 1
                continue

            text = item.text
            if use_to_align_text:
                if to_align_text is not None:
                    if args.warn_on_text_mismatch and item.text and item.text.strip() != to_align_text.strip():
                        warned += 1
                        if warned <= 50:
                            print(
                                f"warn text mismatch line={item.line_num} src={item.src_audio}",
                                file=sys.stderr,
                            )
                    text = to_align_text
                else:
                    # No to_align sentence found for this row.
                    if args.text_source == "to_align":
                        failed += 1
                        continue

            if not text or not item.src_audio:
                skipped += 1
                continue

            # Determine source audio: prefer replaced_audio_path if present.
            if item.replaced_audio_path:
                src = Path(item.replaced_audio_path).expanduser().resolve()
                start = None
                end = None
            else:
                src = Path(item.src_audio)
                if not src.is_absolute():
                    src = (audio_root / item.src_audio).resolve()
                start = item.start
                end = item.end

            if not src.is_file():
                failed += 1
                continue

            if (start is None or end is None) and not item.replaced_audio_path:
                if args.skip_missing_times:
                    skipped += 1
                    continue
                failed += 1
                continue

            out_name = f"{kept:08d}.wav"
            out_wav = wavs_dir / out_name

            try:
                _ffmpeg_cut_resample(
                    src_wav=src,
                    out_wav=out_wav,
                    start=start,
                    end=end,
                    sample_rate=args.sample_rate,
                )
            except Exception:
                failed += 1
                continue

            meta.write(f"{out_name}|{text}\n")

            seg.write(
                json.dumps(
                    {
                        "out_wav": f"wavs/{out_name}",
                        "cutlist_line": item.line_num,
                        "src_audio": item.src_audio,
                        "start": start,
                        "end": end,
                        "replaced_audio_path": item.replaced_audio_path,
                        "text": text,
                        "text_source": ("to_align" if use_to_align_text else "cutlist"),
                        "to_align_sentence_idx": to_align_idx,
                        "cutlist_text": item.text,
                        "to_align_text": to_align_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

            if args.limit and kept >= args.limit:
                break

    if warned:
        print(f"warned_text_mismatch={warned}", file=sys.stderr)
    print(f"done kept={kept} skipped={skipped} failed={failed} out={out_dataset}")
    return 0 if kept > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
