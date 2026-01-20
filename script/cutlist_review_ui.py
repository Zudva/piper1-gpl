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
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    logging.info(msg)


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

    audio_search_roots = []
    for root in [resolved_audio_root, resolved_audio_root.parent, workspace_root]:
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

    def _run_transcription(audio_path, start_s, end_s):
        _log(f"Transcribe requested. audio_path='{audio_path}' start={start_s} end={end_s}")
        if not audio_path:
            _log("Transcribe skipped: empty audio_path")
            return "No audio file"
        if not Path(audio_path).is_file():
            _log(f"Transcribe failed: file not found: {audio_path}")
            return "Audio file not found"
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            _log("Transcribe failed: faster_whisper not installed")
            return "Error: faster_whisper not installed"
            
        if "model" not in model_cache:
            _log("Loading Whisper model (large-v3, cuda/float16)...")
            # Assuming CUDA available; fallback handled in faster_whisper typically requires config check
            # but user has RTX 3090, so we default to cuda.
            model_cache["model"] = WhisperModel("large-v3", device="cuda", compute_type="float16")
        
        target_file = audio_path
        temp_obj = None

        try:
            if start_s is not None or end_s is not None:
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
            model = model_cache["model"]
            segments, _ = model.transcribe(target_file, language="ru")
            text = " ".join([s.text for s in segments]).strip()
            _log(f"Transcription done. chars={len(text)}")
            return text
        except Exception as e:
            _log(f"Transcribe failed: {e}")
            return f"Error: {e}"
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
            # Check previous for start
            if start_s is None and idx > 0:
                prev = rows[idx - 1]
                if prev.get("src_audio") == src_audio and prev.get("end") is not None:
                    start_s = _to_float(prev.get("end"))
                    is_guessed_times = True
            
            # Check next for end
            if end_s is None and idx < len(rows) - 1:
                nxt = rows[idx + 1]
                if nxt.get("src_audio") == src_audio and nxt.get("start") is not None:
                    end_s = _to_float(nxt.get("start"))
                    is_guessed_times = True
                    
            # Fallbacks if only one side found or neither
            if start_s is None and end_s is not None:
                start_s = max(0, end_s - 10.0) # guess 10s before
            if end_s is None and start_s is not None:
                end_s = start_s + 10.0 # guess 10s after
                
            _log(f"Smart Gap: start={start_s}, end={end_s} (guessed={is_guessed_times})")

        _log(f"Segment times: start={start_s}, end={end_s}")
        audio_value = _load_audio_value(audio_path, start_s, end_s)
        text = str(r.get("text") or "")
        text_orig = str(r.get("_text_orig") or "")
        
        status = str(r.get("status") or "")
        sim = r.get("sim")
        
        # Format reason info nicely
        reason_info = f"**Status:** `{status}`\n\n**Similarity:** `{sim}`\n\n**File:** `{src_audio}`"
        if is_guessed_times:
            reason_info += f"\\n\\n**⚠️ Times Guessed:** `{start_s:.2f}` - `{end_s:.2f}`"
        
        review = r.get("review") or {}
        verdict = review.get("verdict", "keep")
        note = review.get("note", "")
        
        progress_label = f"### Item {pos + 1} of {len(indices)}"
        
        # Current choice label
        dropdown_val = _get_choices()[pos]

        return (
            progress_label,     # 1
            audio_value,       # 2
            reason_info,       # 3
            text,              # 4
            text_orig,         # 5
            verdict,           # 6
            note,              # 7
            pos,               # 8
            dropdown_val,      # 9 (for dropdown value)
            audio_path,        # 10
            start_s,           # 11
            end_s              # 12
        )

    def _get_ui_update(pos: int, status_msg: str):
        """Returns list of updates for all UI components."""
        _log(f"UI update: pos={pos}, status='{status_msg}'")
        data = _get_item_data(pos) # 12 items
        
        # Unpack
        (progress_label, audio_value, reason_info, text, text_orig, 
         verdict, note, pos_val, dropdown_val, audio_path, start_s, end_s) = data

        new_choices = _get_choices()
        dropdown_update = gr.Dropdown(value=dropdown_val, choices=new_choices)

        return [
            progress_label,
            audio_value,
            reason_info,
            text,
            text_orig,
            verdict,
            note,
            pos_val,
            dropdown_update,
            status_msg,
            audio_path,
            start_s,
            end_s
        ]

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
        # Update review
        r["review"] = {"verdict": verdict, "note": note}
        
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

    # --- UI Layout ---
    css = """
    .gradio-container { max_width: 1200px !important; }
    .compact-row { gap: 10px; }
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
            status_msg = gr.Textbox(label="", interactive=False, scale=1, show_label=False, placeholder="Ready")

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
                    waveform_options=gr.WaveformOptions(
                        show_recording_waveform=False,
                        waveform_color="#9AA0A6",
                        waveform_progress_color="#22C55E"
                    )
                )
                
                with gr.Row():
                     btn_transcribe = gr.Button("✨ Auto-Transcribe (Whisper)", size="sm", variant="secondary")
                
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
            progress_md, audio_player, info_md, 
            text_input, text_orig_disp, verdict_radio, note_input, 
            state_pos, nav_dropdown, status_msg, audio_path_state,
            start_state, end_state
        ]

        # Init
        demo.load(fn=_load_init, inputs=[], outputs=refresh_outputs)

        # Nav
        nav_dropdown.input(fn=_go_to_by_value, inputs=[nav_dropdown], outputs=refresh_outputs)
        btn_prev.click(fn=_prev, inputs=[state_pos], outputs=refresh_outputs)
        btn_next.click(fn=_next, inputs=[state_pos], outputs=refresh_outputs)

        # Actions
        btn_transcribe.click(fn=_run_transcription, inputs=[audio_path_state, start_state, end_state], outputs=[text_input])

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
    ]

    _log(f"Starting UI on http://{args.host}:{args.port}")
    demo.launch(
        server_name=args.host, 
        server_port=args.port, 
        allowed_paths=allowed_paths,
        css=css,
        theme=gr.themes.Soft()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
