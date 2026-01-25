#!/usr/bin/env python3
"""Sync cutlist.jsonl text from to_align.json.

Goal:
- Keep cutlist as the source-of-truth for timings (src_audio/start/end).
- Keep to_align as the source-of-truth for text (sentences per audio).

This script updates ONLY the `text` field in each cutlist row by consuming
sentences from to_align.json in-order per `src_audio`.

Safety:
- Can write in-place with automatic timestamped backup.
- Produces a report summarizing counts and mismatches.

Assumptions:
- For a given src_audio, the order of rows in cutlist corresponds to the order
  of sentences in to_align for that audio.

Local-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Stats:
	total_rows: int = 0
	updated_rows: int = 0
	unchanged_rows: int = 0
	missing_to_align_rows: int = 0
	invalid_rows: int = 0
	mismatched_original_rows: int = 0


def _now_stamp() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_to_align_map(path: Path) -> dict[str, list[str]]:
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


def _iter_cutlist_rows(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line_num, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except Exception:
				rows.append({"_invalid_json": True, "_line_num": line_num, "_raw": line})
				continue
			if not isinstance(row, dict):
				rows.append({"_invalid_row": True, "_line_num": line_num, "_raw": line})
				continue
			row.setdefault("_line_num", line_num)
			rows.append(row)
	return rows


def _write_cutlist(path: Path, rows: list[dict[str, Any]]) -> None:
	def _clean(d: dict[str, Any]) -> dict[str, Any]:
		out: dict[str, Any] = {}
		for k, v in d.items():
			if str(k).startswith("_"):
				continue
			out[k] = v
		return out

	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			if row.get("_invalid_json") or row.get("_invalid_row"):
				raw = row.get("_raw")
				if isinstance(raw, str) and raw.strip():
					f.write(raw.strip() + "\n")
				continue
			f.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")


def _ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--cutlist", required=True, type=Path)
	p.add_argument("--to-align", required=True, type=Path)
	p.add_argument(
		"--in-place",
		action="store_true",
		help="Write changes back to --cutlist. If set, a backup is written.",
	)
	p.add_argument("--out", type=Path, default=None, help="Output cutlist path (when not --in-place)")
	p.add_argument(
		"--backup-dir",
		type=Path,
		default=None,
		help="Where to write timestamped backup when using --in-place (default: <repo>/backups)",
	)
	p.add_argument(
		"--report",
		type=Path,
		default=None,
		help="Where to write a markdown report (default: work/reports next to alignment/)",
	)
	p.add_argument(
		"--store-original",
		action="store_true",
		help="If text changes, store previous text in row.review.original_text (only if absent).",
	)

	args = p.parse_args()

	cutlist = args.cutlist.expanduser().resolve()
	to_align = args.to_align.expanduser().resolve()

	if not cutlist.is_file():
		raise SystemExit(f"Cutlist not found: {cutlist}")
	if not to_align.is_file():
		raise SystemExit(f"to_align not found: {to_align}")

	if args.in_place:
		out_path = cutlist
	else:
		out_path = (args.out or (cutlist.parent / (cutlist.stem + ".synced.jsonl"))).expanduser().resolve()

	backup_path = None
	if args.in_place:
		repo_root = Path(__file__).resolve().parents[1]
		backup_dir = (args.backup_dir or (repo_root / "backups")).expanduser().resolve()
		backup_dir.mkdir(parents=True, exist_ok=True)
		backup_path = backup_dir / f"cutlist_{_now_stamp()}.jsonl"
		backup_path.write_text(cutlist.read_text(encoding="utf-8"), encoding="utf-8")

	to_align_map = _load_to_align_map(to_align)
	rows = _iter_cutlist_rows(cutlist)

	cursor: dict[str, int] = {}
	per_audio_cutlist: dict[str, int] = {}
	per_audio_missing: dict[str, int] = {}
	stats = Stats()
	mismatch_samples: list[dict[str, Any]] = []

	for row in rows:
		if row.get("_invalid_json") or row.get("_invalid_row"):
			stats.invalid_rows += 1
			continue

		stats.total_rows += 1
		src_audio = str(row.get("src_audio") or "").strip()
		if not src_audio:
			stats.invalid_rows += 1
			continue

		per_audio_cutlist[src_audio] = per_audio_cutlist.get(src_audio, 0) + 1

		sents = to_align_map.get(src_audio)
		idx = cursor.get(src_audio, 0)
		cursor[src_audio] = idx + 1

		if not sents or idx >= len(sents):
			stats.missing_to_align_rows += 1
			per_audio_missing[src_audio] = per_audio_missing.get(src_audio, 0) + 1
			continue

		new_text = sents[idx]
		old_text = str(row.get("text") or "")

		if old_text.strip() != new_text.strip():
			stats.updated_rows += 1
			stats.mismatched_original_rows += 1

			if args.store_original:
				review = row.get("review")
				if not isinstance(review, dict):
					review = {}
					row["review"] = review
				if "original_text" not in review and old_text.strip():
					review["original_text"] = old_text

			row["text"] = new_text

			if len(mismatch_samples) < 50:
				mismatch_samples.append(
					{
						"line": row.get("_line_num"),
						"src_audio": src_audio,
						"idx": idx,
						"old_len": len(old_text),
						"new_len": len(new_text),
						"old": old_text[:160],
						"new": new_text[:160],
					}
				)
		else:
			stats.unchanged_rows += 1

	per_audio_extra: dict[str, int] = {}
	for src_audio, sents in to_align_map.items():
		used = cursor.get(src_audio, 0)
		extra = max(0, len(sents) - used)
		if extra:
			per_audio_extra[src_audio] = extra

	_ensure_parent(out_path)
	_write_cutlist(out_path, rows)

	if args.report is not None:
		report_path = args.report.expanduser().resolve()
	else:
		reports_dir = cutlist.parent.parent / "reports"
		report_path = (reports_dir / f"sync_cutlist_text_{_now_stamp()}.md").resolve()

	_ensure_parent(report_path)

	lines: list[str] = []
	lines.append("# Sync cutlist text from to_align\n\n")
	lines.append(f"- cutlist: `{cutlist}`\n")
	lines.append(f"- to_align: `{to_align}`\n")
	lines.append(f"- output: `{out_path}`\n")
	if backup_path:
		lines.append(f"- backup: `{backup_path}`\n")

	lines.append("\n## Summary\n")
	lines.append(f"- total_rows: {stats.total_rows}\n")
	lines.append(f"- updated_rows: {stats.updated_rows}\n")
	lines.append(f"- unchanged_rows: {stats.unchanged_rows}\n")
	lines.append(f"- missing_to_align_rows: {stats.missing_to_align_rows}\n")
	lines.append(f"- invalid_rows: {stats.invalid_rows}\n")
	lines.append(f"- audio_files_in_to_align: {len(to_align_map)}\n")

	lines.append("\n## Per-audio counts (cutlist vs to_align)\n")
	lines.append("audio_path | cutlist_rows | to_align_sentences | missing | extra\n")
	lines.append("---|---:|---:|---:|---:\n")
	for audio_path in sorted(set(per_audio_cutlist) | set(to_align_map)):
		c = per_audio_cutlist.get(audio_path, 0)
		t = len(to_align_map.get(audio_path, []))
		m = per_audio_missing.get(audio_path, 0)
		e = per_audio_extra.get(audio_path, 0)
		lines.append(f"{audio_path} | {c} | {t} | {m} | {e}\n")

	if mismatch_samples:
		lines.append("\n## First mismatches (sample)\n")
		for s in mismatch_samples:
			lines.append(
				f"- line={s['line']} src={s['src_audio']} idx={s['idx']} len {s['old_len']}→{s['new_len']}\n"
			)

	report_path.write_text("".join(lines), encoding="utf-8")

	print(
		f"done total={stats.total_rows} updated={stats.updated_rows} unchanged={stats.unchanged_rows} "
		f"missing_to_align={stats.missing_to_align_rows} invalid={stats.invalid_rows} out={out_path} report={report_path}",
		file=sys.stderr,
	)

	return 0 if stats.missing_to_align_rows == 0 else 3


if __name__ == "__main__":
	raise SystemExit(main())

