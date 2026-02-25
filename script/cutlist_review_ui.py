#!/usr/bin/env python3
"""Simple UI to review/edit cutlist entries (Stage B output).

Features:
- Load cutlist JSONL
- Filter by status (unmatched/all)
- Navigate entries (prev/next)
- Edit text, add verdict + note
- Save to a new JSONL without touching the original
- Transcribe audio with Whisper (requires faster-whisper)
- Smart Gap Detection for unmatched items (requires ffmpeg)

Local-only. No paid APIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import subprocess
import tempfile
import os
import io
import html
import base64
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import gradio as gr  # type: ignore

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    logging.info(msg)


def _generate_spectrogram(sr: int, data: np.ndarray):
    """Generates a base64 encoded HTML string of the mel spectrogram with overlay."""
    if data is None or sr is None:
        return None
    try:
        # data is (samples,) or (samples, channels)
        if len(data.shape) > 1:
            y = data.mean(axis=1) # mix to mono
        else:
            y = data
        
        duration = len(y) / sr

        # Compute mel spectrogram
        # fmax=8000 is good for speech
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Create figure with 0 margins to align with audio player
        fig = plt.figure(figsize=(12, 3))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='inferno')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode('utf-8')
        
        # Generate HTML with unique ID and data-duration
        # The JS will look for .spectrogram-container and sync with audio
        html = f"""
        <div class="spectrogram-container" data-duration="{duration}" style="position: relative; width: 100%; height: auto; margin-top: -10px; cursor: crosshair;">
            <img src="data:image/png;base64,{b64_img}" style="width: 100%; display: block; height: 150px; object-fit: cover;" draggable="false" />
            <div class="seek-line" style="position: absolute; top: 0; bottom: 0; left: 0; width: 2px; background-color: white; pointer-events: none; display: none;"></div>
        </div>
        """
        return html
    except Exception as e:
        _log(f"Spectrogram generation failed: {e}")
        return None


try:
    import soundfile as sf  # type: ignore
except ImportError:
    _log("Warning: 'soundfile' library not found. Audio playback will be disabled. Install with: pip install soundfile")
    sf = None
except Exception as e:
    _log(f"Warning: Error importing 'soundfile': {e}")
    sf = None


def _load_cutlist(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # keep original text in memory (not written)
    for r in rows:
        if "_text_orig" not in r:
            r["_text_orig"] = r.get("text")
    return rows


def _dump_cutlist(path: Path, rows: list[dict[str, Any]]) -> None:
    def _clean(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            out[k] = v
        return out

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutlist", required=True, type=Path, help="Path to cutlist JSONL")
    p.add_argument("--audio-root", required=True, type=Path, help="Root dir for src_audio paths")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output reviewed cutlist JSONL (default: <cutlist>.reviewed.jsonl)",
    )
    p.add_argument(
        "--mode",
        choices=["unmatched", "all"],
        default="unmatched",
        help="Which entries to review first (default: unmatched)",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=Path("cutlist_review_ui.log"),
        help="Path to log file (default: ./cutlist_review_ui.log)",
    )
    p.add_argument(
        "--output-dataset-root",
        type=Path,
        default=None,
        help="Root folder where the processed dataset should be saved",
    )
    p.add_argument(
        "--output-dataset-name",
        type=str,
        default=None,
        help="Dataset folder name under output root (e.g., felix_mirage_prepared)",
    )
    p.add_argument(
        "--work-on-copy",
        action="store_true",
        help=(
            "Create/use a working copy of the audio dataset and switch playback to it. "
            "This prevents accidental edits against the original dataset."
        ),
    )
    p.add_argument(
        "--work-copy-dir",
        type=Path,
        default=None,
        help=(
            "Where to create the working dataset copy (default: sibling '<audio_root>__work'). "
            "Only used with --work-on-copy."
        ),
    )
    p.add_argument(
        "--work-copy-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help=(
            "How to create the working copy: 'hardlink' is fast and disk-efficient (Linux), "
            "falls back to copy per-file; 'copy' always copies files."
        ),
    )
    p.add_argument(
        "--refresh-work-copy",
        action="store_true",
        help=(
            "Create a new timestamped working copy even if the default work copy already exists."
        ),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)

    args = p.parse_args()

    log_path = args.log_file.expanduser().resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    _log(f"Logging to {log_path}")

    cutlist = args.cutlist.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    if not cutlist.is_file():
        raise SystemExit(f"Cutlist not found: {cutlist}")
    if not audio_root.is_dir():
        raise SystemExit(f"Audio root not found: {audio_root}")

    # Where to store fixed/trimmed audio segments (does not modify the original dataset).
    # Default: <work_dataset>/audio_fixes/, where work_dataset is the parent of alignment/.
    try:
        work_dataset_root = cutlist.parent.parent
    except Exception:
        work_dataset_root = cutlist.parent
    audio_fixes_root = (work_dataset_root / "audio_fixes").resolve()

    rows = _load_cutlist(cutlist)

    def _find_audio_root(root: Path) -> Path:
        if (root / "wavs").is_dir():
            return root

        def _pick_candidate(cands: list[Path]) -> Path | None:
            if not cands:
                return None
            exact = next((c for c in cands if c.name == "felix_mirage"), None)
            if exact:
                return exact
            preferred = next(
                (c for c in cands if c.name.startswith("felix_mirage") and "prepared" not in c.name),
                None,
            )
            if preferred:
                return preferred
            preferred = next((c for c in cands if c.name.startswith("felix_mirage")), None)
            if preferred:
                return preferred
            if len(cands) == 1:
                return cands[0]
            return None

        # Check children of root (e.g., datasets/*/wavs)
        try:
            children = [p for p in root.iterdir() if p.is_dir() and (p / "wavs").is_dir()]
        except Exception:
            children = []
        picked = _pick_candidate(children)
        if picked:
            return picked

        # Check siblings of root (e.g., datasets/felix_mirage_v2_work -> datasets/felix_mirage)
        parent = root.parent
        try:
            siblings = [p for p in parent.iterdir() if p.is_dir() and (p / "wavs").is_dir()]
        except Exception:
            siblings = []
        picked = _pick_candidate(siblings)
        if picked:
            return picked

        return root

    resolved_audio_root = _find_audio_root(audio_root)
    if resolved_audio_root != audio_root:
        _log(f"Audio root auto-resolved: '{audio_root}' -> '{resolved_audio_root}'")

    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent

    training_datasets_root = (workspace_root / "piper-training" / "datasets").resolve()

    def _is_under(path: Path, root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(root)
        except Exception:
            try:
                path.resolve().relative_to(root)
                return True
            except Exception:
                return False

    def _copy_dataset_tree(src: Path, dst: Path, mode: str) -> None:
        if not src.is_dir():
            raise ValueError(f"Source dataset is not a directory: {src}")
        if dst.exists():
            raise ValueError(f"Destination already exists: {dst}")

        def _copy_or_hardlink(src_path: str, dst_path: str) -> str:
            src_p = Path(src_path)
            dst_p = Path(dst_path)
            dst_p.parent.mkdir(parents=True, exist_ok=True)

            if mode == "hardlink":
                try:
                    os.link(src_p, dst_p)
                    return str(dst_p)
                except Exception:
                    # Fall back to a real copy (permissions / cross-device / FS limitations)
                    pass
            shutil.copy2(src_p, dst_p)
            return str(dst_p)

        _log(f"Creating working dataset copy: '{src}' -> '{dst}' (mode={mode})")
        shutil.copytree(src, dst, symlinks=False, copy_function=_copy_or_hardlink)

    if args.work_on_copy:
        preferred_base = workspace_root / "piper-training" / "datasets"
        base_dir = preferred_base if preferred_base.is_dir() else resolved_audio_root.parent
        if base_dir == preferred_base:
            _log(f"Work copy base dir: '{preferred_base}'")
        else:
            _log(f"Work copy base dir not found, falling back to sibling dir: '{base_dir}'")

        default_work_dir = base_dir / f"{resolved_audio_root.name}__work"
        work_dir = (args.work_copy_dir.expanduser().resolve() if args.work_copy_dir else default_work_dir)
        if args.refresh_work_copy and work_dir.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            work_dir = work_dir.parent / f"{work_dir.name}_{ts}"

        if not work_dir.exists():
            _copy_dataset_tree(resolved_audio_root, work_dir, args.work_copy_mode)
        else:
            _log(f"Using existing working dataset copy: '{work_dir}'")

        resolved_audio_root = work_dir
        _log(f"Audio root switched to working copy: '{resolved_audio_root}'")

    output_dataset_root = args.output_dataset_root.expanduser().resolve() if args.output_dataset_root else None
    output_dataset_name = args.output_dataset_name or ""

    # If user didn't specify where replacements go, default to the current audio dataset *when it's safe*.
    # Safe means either we're on an explicit working copy or the audio dataset itself lives under
    # piper-training/datasets (i.e., already a training/work directory).
    if (args.work_on_copy or _is_under(resolved_audio_root, training_datasets_root)) and output_dataset_root is None and not output_dataset_name:
        output_dataset_root = resolved_audio_root.parent
        output_dataset_name = resolved_audio_root.name

    audio_search_roots = []
    for root in [resolved_audio_root, resolved_audio_root.parent, audio_fixes_root, workspace_root]:
        if root and root.exists() and root not in audio_search_roots:
            audio_search_roots.append(root)

    wav_dirs_cache: list[Path] | None = None

    def _get_wav_dirs() -> list[Path]:
        nonlocal wav_dirs_cache
        if wav_dirs_cache is not None:
            return wav_dirs_cache
        wav_dirs: list[Path] = []
        for root in audio_search_roots:
            try:
                for d in root.rglob("wavs"):
                    if d.is_dir() and d not in wav_dirs:
                        wav_dirs.append(d)
            except Exception as e:
                _log(f"Audio search skipped for root '{root}': {e}")
        wav_dirs_cache = wav_dirs
        _log(f"Audio search roots: {[str(r) for r in audio_search_roots]}")
        _log(f"Audio search wavs dirs: {len(wav_dirs_cache)}")
        return wav_dirs_cache

    def _resolve_audio_path(src_audio: str, audio_path: str) -> str:
        if Path(audio_path).is_file():
            return audio_path
        # Try direct join against search roots
        for root in audio_search_roots:
            candidate = (root / src_audio).resolve()
            if candidate.is_file():
                _log(f"Audio path resolved via root '{root}': {candidate}")
                return str(candidate)
        # Try any wavs directory under search roots
        filename = Path(src_audio).name
        if not filename:
            return audio_path
        for d in _get_wav_dirs():
            candidate = d / filename
            if candidate.is_file():
                _log(f"Audio path resolved via wavs dir '{d}': {candidate}")
                return str(candidate)
        return audio_path
    out_path = (
        args.out.expanduser().resolve()
        if args.out
        else cutlist.with_suffix(cutlist.suffix + ".reviewed.jsonl")
    )

    # build index list
    if args.mode == "unmatched":
        indices = [i for i, r in enumerate(rows) if r.get("status") == "unmatched"]
    else:
        indices = list(range(len(rows)))

    if not indices:
        print("No entries to review for the selected mode.")
        return 0

    # Lazy load model
    model_cache = {}

    def _progress_update(progress, value: float, desc: str) -> None:
        try:
            progress(value, desc=desc)
        except Exception:
            pass
        if desc:
            pct = int(value * 100)
            _log(f"Status: {desc} ({pct}%)")

    def _run_transcription(audio_path, start_s, end_s, progress=gr.Progress()):
        _log(f"Transcribe requested. audio_path='{audio_path}' start={start_s} end={end_s}")
        _progress_update(progress, 0, "Проверка аудио")
        empty_karaoke = (
            "<div style='color:gray'>Нажмите 'Auto-Transcribe' для разметки текущего сэмпла.</div>"
        )
        if not audio_path:
            _log("Transcribe skipped: empty audio_path")
            return "No audio file", empty_karaoke, "Ошибка: пустой путь к аудио"
        if not Path(audio_path).is_file():
            _log(f"Transcribe failed: file not found: {audio_path}")
            return "Audio file not found", empty_karaoke, "Ошибка: файл аудио не найден"
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            _log("Transcribe failed: faster_whisper not installed")
            return "Error: faster_whisper not installed", empty_karaoke, "Ошибка: faster_whisper не установлен"
            
        if "model" not in model_cache:
            _progress_update(progress, 0.15, "Загрузка модели Whisper")
            _log("Loading Whisper model (large-v3, cuda/float16)...")
            # Assuming CUDA available; fallback handled in faster_whisper typically requires config check
            # but user has RTX 3090, so we default to cuda.
            model_cache["model"] = WhisperModel("large-v3", device="cuda", compute_type="float16")
        
        target_file = audio_path
        temp_obj = None

        try:
            if start_s is not None or end_s is not None:
                _progress_update(progress, 0.35, "Нарезка сегмента")
                # Use ffmpeg to cut
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                temp_obj = Path(temp_path)
                
                cmd = ["ffmpeg", "-y", "-i", audio_path]
                if start_s is not None:
                    cmd.extend(["-ss", str(float(start_s))])
                if end_s is not None:
                    cmd.extend(["-to", str(float(end_s))])
                
                # Force resample for consistency
                cmd.extend(["-ar", "16000", "-ac", "1", temp_path])
                
                _log(f"Cutting audio with ffmpeg: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                target_file = temp_path

            _log(f"Transcribing {target_file}...")
            _progress_update(progress, 0.55, "Распознавание речи")
            model = model_cache["model"]
            # Use word_timestamps=True to get word-level timing
            segments, _ = model.transcribe(target_file, language="ru", word_timestamps=True)
            
            full_text_list = []
            html_parts = []
            html_parts.append('<div class="karaoke-text" style="line-height: 1.6; font-size: 1.1rem; padding: 10px; border-radius: 4px;">')
            
            for segment in segments:
                for word in segment.words:
                    w = word.word.strip()
                    full_text_list.append(w)
                    safe_w = html.escape(w)
                    # Add data attributes for JS
                    html_parts.append(f'<span class="karaoke-word" data-start="{word.start:.2f}" data-end="{word.end:.2f}" style="margin-right: 4px; padding: 2px 4px; border-radius: 3px; transition: background 0.1s;">{safe_w}</span>')
            
            html_parts.append('</div>')
            
            text = " ".join(full_text_list).strip()
            karaoke = "".join(html_parts)
            
            _log(f"Transcription done. chars={len(text)}")
            _progress_update(progress, 1.0, "Готово")
            return text, karaoke, "Готово"
        except Exception as e:
            _log(f"Transcribe failed: {e}")
            return f"Error: {e}", empty_karaoke, f"Ошибка: {e}"
        finally:
            if temp_obj and temp_obj.exists():
                try:
                    temp_obj.unlink()
                except:
                    pass

    def _label(i: int) -> str:
        r = rows[i]
        status = r.get("status", "?")
        sim = r.get("sim")
        sim_str = f"{sim:.2f}" if isinstance(sim, (int, float)) else "N/A"
        # Add a visual indicator if reviewed
        verdict = r.get("review", {}).get("verdict")
        mark = "✓" if verdict else " "
        return f"{mark} [{i}] {status} (sim={sim_str})"

    def _clamp_view(audio_path, start_v, end_v, context_pad=0.0):
        """Updates audio player and spectrogram based on start/end inputs without saving."""
        if not audio_path:
            return gr.update(), gr.update()
        
        # Ensure we have valid floats or None
        if start_v == 0 and end_v == 0:
            pass
            
        # Apply padding if requested, but respect file boundaries (0)
        actual_start = start_v
        actual_end = end_v
        
        if start_v is not None:
            actual_start = max(0, float(start_v) - float(context_pad))
        else:
            # If no start defined, context pad means "start from beginning" probably
            # but _load_audio_value processes None as "start of file"
            pass 

        if end_v is not None:
            # If end is defined, extend it
            # But _load_audio_value needs to check file duration to clamp upper bound? 
            # _load_audio_value does checking against total_frames.
            actual_end = float(end_v) + float(context_pad)

        val = _load_audio_value(audio_path, actual_start, actual_end)
        if val is None:
            return gr.update(), gr.update()

        sr, data = val
        spec_html = _generate_spectrogram(sr, data)
        return (sr, data), spec_html

    # We need a dynamic list of choices to update the checkmarks
    def _get_choices():
        return [_label(i) for i in indices]

    def _load_audio_value(audio_path: str, start_s: float | None, end_s: float | None):
        if not audio_path:
            _log("Audio load skipped: empty audio_path")
            return None
        if sf is None:
            _log("Audio load failed: soundfile not loaded")
            return None
        path_obj = Path(audio_path)
        if not path_obj.is_file():
            _log(f"Audio load failed: file not found: {audio_path}")
            return None
        try:
            info = sf.info(audio_path)
            sr = info.samplerate
            total_frames = info.frames
            start_sample = 0
            end_sample = total_frames
            if start_s is not None:
                start_sample = max(0, int(start_s * sr))
            if end_s is not None:
                end_sample = min(total_frames, int(end_s * sr))
            frames = max(0, end_sample - start_sample)
            if frames <= 0:
                _log(
                    f"Audio load failed: empty segment start={start_s} end={end_s} frames={frames}"
                )
                return None
            data, sr = sf.read(audio_path, start=start_sample, frames=frames, always_2d=False)
            if getattr(data, "dtype", None) is not None and data.dtype != "float32":
                try:
                    data = data.astype("float32", copy=False)
                except Exception:
                    pass
            _log(
                f"Audio loaded: sr={sr}, frames={frames}, shape={getattr(data, 'shape', None)}"
            )
            return (sr, data)
        except Exception as e:
            _log(f"Audio load failed: {audio_path}: {e}")
            return None

    def _get_item_data(pos: int):
        """Returns raw data tuple for the item."""
        pos = max(0, min(pos, len(indices) - 1))
        idx = indices[pos]
        r = rows[idx]
        
        src_audio = str(r.get("src_audio") or "")
        if Path(src_audio).is_absolute():
            audio_path = src_audio
        else:
            audio_path = str((resolved_audio_root / src_audio).resolve())
        audio_path = _resolve_audio_path(src_audio, audio_path)
        _log(
            f"Item pos={pos}, idx={idx}, src_audio='{src_audio}', audio_path='{audio_path}'"
        )
        def _to_float(v: Any) -> float | None:
            try:
                return float(v)
            except Exception:
                return None

        start_s = _to_float(r.get("start"))
        end_s = _to_float(r.get("end"))

        # Smart Gap Detection for Unmatched items
        is_guessed_times = False
        if start_s is None or end_s is None:
            # Check previous for start (scan backwards)
            if start_s is None:
                curr_i = idx - 1
                while curr_i >= 0:
                    prev = rows[curr_i]
                    # If file changes, we can't use this as start reference
                    if prev.get("src_audio") != src_audio:
                        break
                    # If we find a valid end, that's our start
                    if prev.get("end") is not None:
                        start_s = _to_float(prev.get("end"))
                        is_guessed_times = True
                        break
                    curr_i -= 1

            # Check next for end (scan forwards)
            if end_s is None:
                curr_i = idx + 1
                while curr_i < len(rows):
                    nxt = rows[curr_i]
                    if nxt.get("src_audio") != src_audio:
                        break
                    if nxt.get("start") is not None:
                        end_s = _to_float(nxt.get("start"))
                        is_guessed_times = True
                        break
                    curr_i += 1

            # Fallbacks if only one side found or neither
            if start_s is None and end_s is not None:
                start_s = max(0, end_s - 10.0)  # guess 10s before
            if end_s is None and start_s is not None:
                end_s = start_s + 10.0  # guess 10s after

            _log(f"Smart Gap: start={start_s}, end={end_s} (guessed={is_guessed_times})")

        _log(f"Segment times: start={start_s}, end={end_s}")

        review = r.get("review") or {}
        replaced_audio = review.get("replaced_audio") if isinstance(review, dict) else None
        if isinstance(replaced_audio, dict):
            override_path = replaced_audio.get("new_audio_path")
            if override_path:
                audio_path = str(override_path)

        audio_value = _load_audio_value(audio_path, start_s, end_s)
        
        # Generate spectrogram
        spectrogram_fig = None
        if audio_value is not None:
             spectrogram_fig = _generate_spectrogram(audio_value[0], audio_value[1])

        text = str(r.get("text") or "")
        text_orig = str(r.get("_text_orig") or "")
        karaoke_default = (
            "<div style='color:gray'>Нажмите 'Auto-Transcribe' для разметки текущего сэмпла.</div>"
        )
        
        status = str(r.get("status") or "")
        sim = r.get("sim")
        
        # Format reason info nicely
        reason_info = f"**Status:** `{status}`\n\n**Similarity:** `{sim}`\n\n**File:** `{src_audio}`"
        if is_guessed_times:
            reason_info += f"\\n\\n**⚠️ Times Guessed:** `{start_s:.2f}` - `{end_s:.2f}`"
        if isinstance(review, dict) and review.get("replaced_audio"):
            rep = review.get("replaced_audio")
            if isinstance(rep, dict):
                rep_src = rep.get("new_src_audio") or rep.get("new_audio_path")
                if rep_src:
                    reason_info += f"\\n\\n**🔁 Replaced:** `{rep_src}`"
        
        review = r.get("review") or {}
        verdict = review.get("verdict", "keep")
        note = review.get("note", "")
        
        progress_label = f"### Item {pos + 1} of {len(indices)}"
        
        # Current choice label
        dropdown_val = _get_choices()[pos]

        return (
            progress_label,     # 1
            audio_value,       # 2
            spectrogram_fig,   # 3 (NEW)
            reason_info,       # 4
            text,              # 5
            text_orig,         # 6
            karaoke_default,   # 7
            verdict,           # 8
            note,              # 9
            pos,               # 10
            dropdown_val,      # 11
            audio_path,        # 12
            start_s,           # 13
            end_s              # 14
        )

    def _get_ui_update(pos: int, status_msg: str):
        """Returns list of updates for all UI components."""
        _log(f"UI update: pos={pos}, status='{status_msg}'")
        data = _get_item_data(pos) 
        
        # Unpack
        (progress_label, audio_value, spectrogram_fig, reason_info, text, text_orig,
         karaoke_default, verdict, note, pos_val, dropdown_val, audio_path, start_s, end_s) = data

        new_choices = _get_choices()
        dropdown_update = gr.Dropdown(value=dropdown_val, choices=new_choices)

        return [
            progress_label,
            audio_value,
            spectrogram_fig,
            reason_info,
            text,
            text_orig,
            karaoke_default,
            verdict,
            note,
            pos_val,
            dropdown_update,
            status_msg,
            audio_path,
            start_s,
            end_s,
            start_s,
            end_s
        ]

    def _set_verdict(pos: int, verdict: str, note: str):
        idx = indices[pos]
        r = rows[idx]
        review = r.get("review")
        if not isinstance(review, dict):
            review = {}
        review["verdict"] = verdict
        if note is not None:
            review["note"] = note
        r["review"] = review
        return _get_ui_update(pos, f"Verdict set: {verdict}")

    def _drop_current(pos: int, note: str):
        return _set_verdict(pos, "drop", note)

    def _save_edit_logic(pos: int, text: str, verdict: str, note: str):
        _log(f"Save edit: pos={pos}, verdict='{verdict}'")
        idx = indices[pos]
        r = rows[idx]
        # Update text
        if text is not None:
            r["text"] = text
            if r.get("_text_orig") != text and "text_orig" not in r:
                r["text_orig"] = r.get("_text_orig")
            else:
                pass
        # Update review (preserve existing fields like replaced_audio)
        review = r.get("review")
        if not isinstance(review, dict):
            review = {}
        review["verdict"] = verdict
        review["note"] = note
        r["review"] = review
        
    def _save_and_stay(pos: int, text: str, verdict: str, note: str):
        _save_edit_logic(pos, text, verdict, note)
        return _get_ui_update(pos, "Saved (Stayed)")

    def _save_and_next(pos: int, text: str, verdict: str, note: str):
        _save_edit_logic(pos, text, verdict, note)
        new_pos = pos + 1
        if new_pos >= len(indices):
             new_pos = pos # stay at end
        return _get_ui_update(new_pos, f"Saved & Moved to {new_pos+1}")

    def _go_to_by_value(val: str):
        if not val:
             return _get_ui_update(0, "")
        
        m = re.search(r"\[(\d+)\]", val)
        if not m:
            return _get_ui_update(0, "")
        
        idx = int(m.group(1))
        try:
            pos = indices.index(idx)
        except ValueError:
            pos = 0
        return _get_ui_update(pos, "")

    def _prev(pos: int):
        new_pos = max(0, pos - 1)
        return _get_ui_update(new_pos, "")
    
    def _next(pos: int):
        new_pos = min(len(indices) - 1, pos + 1)
        return _get_ui_update(new_pos, "")

    def _load_init():
        _log("UI init load")
        return _get_ui_update(0, "Loaded")

    def _save_file():
        _dump_cutlist(out_path, rows)
        _log(f"Saved JSONL to: {out_path}")
        return f"Saved JSONL to: {out_path}"

    def _replace_audio_segment(pos: int, audio_path: str, start_s: float | None, end_s: float | None):
        if start_s is None or end_s is None:
            return _get_ui_update(pos, "Ошибка: нет start/end для сегмента")
        if not audio_path or not Path(audio_path).is_file():
            return _get_ui_update(pos, "Ошибка: аудио файл не найден")

        idx = indices[pos]
        r = rows[idx]
        orig_src_audio = str(r.get("src_audio") or "")
        orig_audio_path = str(audio_path)

        start_ms = int(float(start_s) * 1000)
        end_ms = int(float(end_s) * 1000)
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Store under: <audio_fixes_root>/<parent>/<stem>/seg_<...>__<ts>/segment.wav
        parent_rel = Path(orig_src_audio).parent.as_posix() if orig_src_audio else "unknown"
        stem = Path(orig_src_audio).stem if orig_src_audio else Path(audio_path).stem
        fix_dir = (audio_fixes_root / parent_rel / stem / f"seg_{start_ms}ms_{end_ms}ms__{ts}").resolve()
        fix_dir.mkdir(parents=True, exist_ok=True)

        out_path = fix_dir / "segment.wav"

        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(float(start_s)),
            "-to", str(float(end_s)),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(out_path)
        ]
        _log(f"Replacing audio with ffmpeg: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        review = r.get("review") or {}
        if not isinstance(review, dict):
            review = {}
        review["replaced_audio"] = {
            "orig_src_audio": orig_src_audio,
            "orig_audio_path": orig_audio_path,
            "new_audio_path": str(out_path),
            "fix_dir": str(fix_dir),
            "start": float(start_s),
            "end": float(end_s),
            "created_at": ts,
            "ffmpeg_cmd": cmd,
        }
        r["review"] = review

        (fix_dir / "info.json").write_text(
            json.dumps(
                {
                    "orig_src_audio": orig_src_audio,
                    "orig_audio_path": orig_audio_path,
                    "start": float(start_s),
                    "end": float(end_s),
                    "new_audio_path": str(out_path),
                    "created_at": ts,
                    "ffmpeg_cmd": cmd,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        _log(f"Replaced audio (stored): {orig_src_audio} -> {out_path}")
        return _get_ui_update(pos, f"Replaced audio stored: {out_path}")

    # --- UI Layout ---
    css = """
    .active-word { background-color: #fde047; color: #000; font-weight: bold; box-shadow: 0 0 2px rgba(0,0,0,0.2); }

    /* Allow karaoke text to wrap instead of horizontal scrolling */
    #karaoke_box, #karaoke_box .karaoke-text, #karaoke_box .karaoke-word {
        white-space: normal !important;
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    /* Keep karaoke background consistent with dark theme */
    #karaoke_box, #karaoke_box .karaoke-text,
    #karaoke_box .gradio-html, #karaoke_box .prose,
    #karaoke_box > div, #karaoke_box .wrap, #karaoke_box .container {
        background: transparent !important;
        color: inherit !important;
    }
    /* Gradio block container (avoid forced single-line in some themes) */
    .block.svelte-1plpy97.padded.auto-margin {
        white-space: normal !important;
    }

    .gradio-container { max_width: 1200px !important; }
    .compact-row { gap: 10px; }
    /* Hide Share button in local UI (Gradio versions without show_share_button) */
    #share-btn, .share-button, button[aria-label="Share"], button[title="Share"] { display: none !important; }
    """
    
    js_sync = """
    <script>
    function setupAudioSync() {
        // Poll because Gradio might replace elements dynamically
        setInterval(() => {
            const audio = document.querySelector('#main_audio audio') || document.querySelector('audio');
            const container = document.querySelector('.spectrogram-container');
            const karaokeBox = document.querySelector('#karaoke_box');
            const statusInput = document.querySelector('#status_msg textarea') || document.querySelector('#status_msg input');
            const statusBox = document.querySelector('#status_msg');
            if (statusInput) {
                const currStatus = statusInput.value || statusInput.textContent || '';
                if (window.__lastStatusMsg !== currStatus) {
                    window.__lastStatusMsg = currStatus;
                    if (currStatus) {
                        console.log('[cutlist-ui] status:', currStatus);
                    }
                }
            }
            if (statusBox && !statusBox._statusObserver) {
                const observer = new MutationObserver(() => {
                    const input = statusBox.querySelector('textarea') || statusBox.querySelector('input');
                    const txt = (input && (input.value || input.textContent)) || statusBox.textContent || '';
                    if (txt && window.__lastStatusMsg !== txt) {
                        window.__lastStatusMsg = txt;
                        console.log('[cutlist-ui] status:', txt);
                    }
                });
                observer.observe(statusBox, { childList: true, subtree: true, characterData: true });
                statusBox._statusObserver = observer;
            }
            
            if (audio) {
                // Determine if we need to hook (or re-hook if elements changed)
                const isHooked = audio._isSyncHooked;

                // --- CLICK TO SEEK (Check if we need to attach onclick) ---
                if (container && !container._isClickHooked) {
                    container.onclick = (e) => {
                        const rect = container.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const w = rect.width;
                        if (w > 0) {
                             const duration = parseFloat(container.dataset.duration) || audio.duration;
                             if (duration > 0) {
                                 const seekTime = (x / w) * duration;
                                 if (Number.isFinite(seekTime)) {
                                     audio.currentTime = seekTime;
                                     audio.play(); // Optional: play on seek
                                 }
                             }
                        }
                    };
                    container._isClickHooked = true;
                }
                
                // If hooked, we still run the update loop via animation frame, 
                // but we also need to efficiently update karaoke even if we are "hooked".
                // Actually, the requestAnimationFrame loop handles it.
                
                if (isHooked) return;
                
                let animFrame;
                
                const update = () => {
                   if (audio.paused || audio.ended) return;

                   const curr = audio.currentTime;
                   
                   // 1. Spectrogram Sync
                   if (container && document.contains(container)) {
                       const line = container.querySelector('.seek-line');
                       const duration = parseFloat(container.dataset.duration);
                       const d = duration || audio.duration;
                       if (line && d > 0) {
                           const pct = (curr / d) * 100;
                           line.style.left = pct + '%';
                           line.style.display = 'block';
                       }
                   }
                   
                   // 2. Karaoke Sync
                   if (karaokeBox && document.contains(karaokeBox)) {
                       // Optimize: Select all spans once? No, Gradio replaces innerHTML on update.
                       // We can query selector inside the loop if list is small (~20 words).
                       const words = karaokeBox.querySelectorAll('.karaoke-word');
                       words.forEach(w => {
                           const start = parseFloat(w.dataset.start);
                           const end = parseFloat(w.dataset.end);
                           // Simple range check
                           if (curr >= start && curr <= end) {
                               w.classList.add('active-word');
                           } else {
                               w.classList.remove('active-word');
                           }
                       });
                   }

                   animFrame = requestAnimationFrame(update);
                };
                
                audio.addEventListener('play', () => {
                    cancelAnimationFrame(animFrame);
                    update();
                });
                
                audio.addEventListener('pause', () => {
                    cancelAnimationFrame(animFrame);
                });
                
                audio.addEventListener('ended', () => {
                    cancelAnimationFrame(animFrame);
                });
                
                audio.addEventListener('seeked', () => {
                   // One-shot update for seek
                   const curr = audio.currentTime;
                   // Spectrogram
                   if (container) {
                       const line = container.querySelector('.seek-line');
                       const duration = parseFloat(container.dataset.duration);
                       const d = duration || audio.duration;
                       if (line && d > 0) {
                           line.style.left = ((curr/d)*100) + '%';
                       }
                   }
                   // Karaoke
                   if (karaokeBox) {
                       const words = karaokeBox.querySelectorAll('.karaoke-word');
                       words.forEach(w => {
                           const start = parseFloat(w.dataset.start);
                           const end = parseFloat(w.dataset.end);
                           if (curr >= start && curr <= end) {
                               w.classList.add('active-word');
                           } else {
                               w.classList.remove('active-word');
                           }
                       });
                   }
                });
                
                audio._isSyncHooked = true;
                
                if (!audio.paused) update();
            }
        }, 500);
    }
    // Run on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupAudioSync);
    } else {
        setupAudioSync();
    }
    </script>
    """

    with gr.Blocks(title="Piper Cutlist Review") as demo:
        
        # State
        state_pos = gr.State(0)
        audio_path_state = gr.State("")
        start_state = gr.State(None)
        end_state = gr.State(None)

        # Header
        with gr.Row():
            gr.Markdown(f"# ✂️ Piper Data Review\\n**File:** `{cutlist.name}` | **Mode:** `{args.mode}`")
            btn_write_file = gr.Button("💾 Write to Disk (Save JSONL)", variant="secondary", scale=0)
            status_msg = gr.Textbox(label="", interactive=False, scale=1, show_label=False, placeholder="Ready", elem_id="status_msg")

        with gr.Row():
            # LEFT COL: Navigation
            with gr.Column(scale=1):
                progress_md = gr.Markdown("### Loading...")
                
                nav_dropdown = gr.Dropdown(
                    choices=_get_choices(), 
                    label="Jump to entry", 
                    value=_get_choices()[0] if indices else None,
                    interactive=True
                )
                
                with gr.Row():
                    btn_prev = gr.Button("⬅️ Prev")
                    btn_next = gr.Button("Next ➡️")

                info_md = gr.Markdown("Item Info")

            # RIGHT COL: Editor
            with gr.Column(scale=2):
                audio_player = gr.Audio(
                    label="Audio Segment",
                    type="numpy",
                    autoplay=True,
                    elem_id="main_audio",
                    waveform_options=gr.WaveformOptions(
                        show_recording_waveform=False,
                        waveform_color="#9AA0A6",
                        waveform_progress_color="#22C55E"
                    )
                )
                
                spectrogram_html = gr.HTML(label="Mel Spectrogram", elem_id="spectrogram_box")
                
                with gr.Accordion("Karaoke / Alignment View", open=True):
                    karaoke_html = gr.HTML(label="Karaoke", value="<div style='color:gray'>Click 'Auto-Transcribe' to see word alignment here.</div>", elem_id="karaoke_box")
                
                with gr.Row():
                     btn_transcribe = gr.Button("✨ Auto-Transcribe (Whisper)", size="sm", variant="secondary")

                with gr.Accordion("Trim / Replace (backup)", open=True):
                    gr.Markdown(
                        "Трим берёт границы **start/end** из текущей строки cutlist (или из Smart Gap для `unmatched`).\n"
                        "Спектрограмма сейчас только для просмотра (выделение мышкой не реализовано).\n\n"
                        "Кнопка ниже сохранит сегмент как **backup** и запишет путь в `review.replaced_audio.new_audio_path`.\n\n"
                        f"Папка для исправлений: `{audio_fixes_root}`"
                    )
                    with gr.Row():
                        start_input = gr.Number(label="start (sec)", precision=3)
                        end_input = gr.Number(label="end (sec)", precision=3)
                        btn_clamp = gr.Button("🔍 Focus", variant="secondary", scale=0)
                        btn_context = gr.Button("🔍 +2s", variant="secondary", scale=0)
                    with gr.Row():
                        btn_drop = gr.Button("🗑️ Drop segment (exclude from training)", size="sm", variant="secondary")
                    btn_replace_audio = gr.Button(
                        "✂️ Trim/Replace: сохранить сегмент (backup)",
                        size="sm",
                        variant="primary",
                        interactive=True,
                    )
                
                with gr.Accordion("Original Text (Reference)", open=False):
                    text_orig_disp = gr.Textbox(label="Original", interactive=False, lines=2)
                
                text_input = gr.Textbox(label="Transcribed/Corrected Text", lines=3, placeholder="Edit text here if needed...")
                
                with gr.Group():
                    gr.Markdown("### Verdict")
                    with gr.Row():
                        verdict_radio = gr.Radio(
                            choices=["keep", "drop", "skip"], 
                            label="Decision", 
                            show_label=False,
                            container=False
                        )
                    note_input = gr.Textbox(label="Notes (Optional)", placeholder="Reason for drop/edit...", lines=1)

                with gr.Row():
                    btn_save_stay = gr.Button("Save", variant="secondary")
                    btn_save_next = gr.Button("Save & Next ➡️", variant="primary")

        # --- Event Wiring ---
        
        # Common outputs for page refreshes
        refresh_outputs = [
            progress_md, audio_player, spectrogram_html, info_md,
            text_input, text_orig_disp, karaoke_html, verdict_radio, note_input,
            state_pos, nav_dropdown, status_msg, audio_path_state,
            start_state, end_state,
            start_input, end_input
        ]

        # Init
        demo.load(fn=_load_init, inputs=[], outputs=refresh_outputs)

        # Nav
        nav_dropdown.input(fn=_go_to_by_value, inputs=[nav_dropdown], outputs=refresh_outputs)
        btn_prev.click(fn=_prev, inputs=[state_pos], outputs=refresh_outputs)
        btn_next.click(fn=_next, inputs=[state_pos], outputs=refresh_outputs)

        # Actions
        btn_transcribe.click(
            fn=_run_transcription,
            inputs=[audio_path_state, start_input, end_input],
            outputs=[text_input, karaoke_html, status_msg]
        )

        btn_clamp.click(
            fn=_clamp_view,
            inputs=[audio_path_state, start_input, end_input],
            outputs=[audio_player, spectrogram_html]
        )
        
        # Context button adds 2s padding (on both sides)
        btn_context.click(
            fn=lambda p, s, e: _clamp_view(p, s, e, 2.0),
            inputs=[audio_path_state, start_input, end_input],
            outputs=[audio_player, spectrogram_html]
        )

        btn_replace_audio.click(
            fn=_replace_audio_segment,
            inputs=[state_pos, audio_path_state, start_input, end_input],
            outputs=refresh_outputs
        )

        btn_drop.click(
            fn=_drop_current,
            inputs=[state_pos, note_input],
            outputs=refresh_outputs,
        )

        btn_save_stay.click(
            fn=_save_and_stay,
            inputs=[state_pos, text_input, verdict_radio, note_input],
            outputs=refresh_outputs
        )
        
        btn_save_next.click(
            fn=_save_and_next,
            inputs=[state_pos, text_input, verdict_radio, note_input],
            outputs=refresh_outputs
        )

        # File IO
        btn_write_file.click(fn=_save_file, inputs=[], outputs=[status_msg])

    allowed_paths = [
        str(audio_root.resolve()),
        str((audio_root / "wavs").resolve()),
        str(audio_fixes_root.resolve()),
    ]

    _log(f"Starting UI on http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host, 
        server_port=args.port, 
        allowed_paths=allowed_paths,
        css=css,
        head=js_sync,
        theme=gr.themes.Soft()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
