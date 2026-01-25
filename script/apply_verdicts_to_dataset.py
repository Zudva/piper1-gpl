#!/usr/bin/env python3
"""Create a filtered dataset based on listen_and_verify verdicts.

Typical flow:
  1) Listen + mark:
     python script/listen_and_verify.py <low_similarity_list.tsv> --player paplay

  2) Filter dataset:
     python script/apply_verdicts_to_dataset.py \
       --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
       --verified <.../low_similarity_list_verified.tsv>

This script is local-only and does not use paid APIs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def _read_verified(path: Path) -> dict[str, str]:
    """Return mapping wav -> verdict."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"wav", "verdict"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f"Verified TSV must contain columns {sorted(required)}; got {reader.fieldnames}")
        out: dict[str, str] = {}
        for row in reader:
            wav = (row.get("wav") or "").strip()
            verdict = (row.get("verdict") or "").strip().lower()
            if not wav:
                continue
            out[wav] = verdict
        return out


def _iter_metadata_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            yield line


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path, help="Input dataset root (config.json, wavs/, metadata_2col.csv)")
    p.add_argument("--verified", required=True, type=Path, help="TSV output from listen_and_verify.py (*_verified.tsv)")
    p.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Output dataset dir (default: <dataset>_filtered_<ts>)",
    )
    p.add_argument(
        "--drop",
        default="bad,missing",
        help="Comma-separated verdicts to drop (default: bad,missing). Example: bad,missing,skip",
    )
    p.add_argument(
        "--wavs-mode",
        choices=["symlink_dir"],
        default="symlink_dir",
        help="How to provide wavs in output dataset (default: symlink_dir).",
    )

    args = p.parse_args()

    dataset = args.dataset.expanduser().resolve()
    verified = args.verified.expanduser().resolve()

    if not dataset.is_dir():
        raise SystemExit(f"Dataset dir not found: {dataset}")
    meta_in = dataset / "metadata_2col.csv"
    cfg_in = dataset / "config.json"
    wavs_in = dataset / "wavs"
    if not meta_in.is_file():
        raise SystemExit(f"Missing metadata_2col.csv: {meta_in}")
    if not cfg_in.is_file():
        raise SystemExit(f"Missing config.json: {cfg_in}")
    if not wavs_in.is_dir():
        raise SystemExit(f"Missing wavs/: {wavs_in}")
    if not verified.is_file():
        raise SystemExit(f"Verified TSV not found: {verified}")

    drop = {v.strip().lower() for v in (args.drop or "").split(",") if v.strip()}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dataset = (args.output_dataset.expanduser().resolve() if args.output_dataset else Path(f"{dataset}_filtered_{ts}").resolve())

    out_dataset.mkdir(parents=True, exist_ok=False)

    # Copy config.json
    (out_dataset / "config.json").write_text(cfg_in.read_text(encoding="utf-8"), encoding="utf-8")

    # wavs/
    if args.wavs_mode == "symlink_dir":
        (out_dataset / "wavs").symlink_to(wavs_in)

    # Filter metadata
    verdict_by_wav = _read_verified(verified)

    kept_lines: list[str] = []
    dropped_lines: list[str] = []
    unknown = 0

    for line in _iter_metadata_lines(meta_in):
        wav = line.split("|", 1)[0].strip()
        verdict = verdict_by_wav.get(wav)
        if verdict is None:
            unknown += 1
            kept_lines.append(line)
            continue
        if verdict in drop:
            dropped_lines.append(line)
        else:
            kept_lines.append(line)

    (out_dataset / "metadata_2col.csv").write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    print(f"Input dataset:   {dataset}")
    print(f"Verified TSV:    {verified}")
    print(f"Output dataset:  {out_dataset}")
    print(f"Drop verdicts:   {sorted(drop)}")
    print(f"Metadata lines:  in={len(kept_lines)+len(dropped_lines)} kept={len(kept_lines)} dropped={len(dropped_lines)}")
    print(f"Not in verified: {unknown} (kept)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
