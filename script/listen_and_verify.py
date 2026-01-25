#!/usr/bin/env python3
"""
Interactive audio verification tool for Whisper validation issues.
Plays audio files and lets you mark them as OK or BAD.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import wave
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


def _wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def play_audio(wav_path: Path) -> bool:
    """Play audio file and ensure we wait for (approximately) its full duration.

    Some players/drivers return early even though audio is still buffered. To avoid
    “cutting off” perceived playback, we compute WAV duration and wait for it.
    If a backend exits too early, we try the next one.
    """

    return play_audio_with_backend(wav_path, backend="auto")


def play_audio_with_backend(wav_path: Path, backend: str = "auto") -> bool:
    """Play audio using a selected backend.

    backend:
      - auto: try best available order
      - pw-play, paplay, ffplay, aplay
      - ffmpeg-aplay: decode via ffmpeg and play via aplay (raw PCM)
    """

    try:
        duration_s = _wav_duration_seconds(wav_path)
    except Exception:
        duration_s = 0.0

    # If a backend returns far earlier than the file duration, consider it broken.
    min_ok_runtime = max(0.25, duration_s * 0.90)

    def _run_and_wait(cmd: List[str], allow_early_ok: bool = False) -> bool:
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False

        # Wait for player to finish; if it exits suspiciously early, treat as failure.
        try:
            # For short clips we still allow a safety margin.
            proc.wait(timeout=max(1.0, duration_s + 3.0) if duration_s > 0 else None)
        except subprocess.TimeoutExpired:
            # Player is hanging; terminate.
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            return False

        runtime = time.monotonic() - start

        # If the player returned early but the WAV is longer, wait out remaining time
        # (some stacks buffer audio and return early; waiting avoids perceived cut-offs).
        if duration_s > 0 and runtime < duration_s:
            time.sleep(max(0.0, duration_s - runtime) + 0.10)

        if proc.returncode != 0:
            return False
        if not allow_early_ok and duration_s > 0 and runtime < min_ok_runtime:
            return False
        return True

    def _ffmpeg_aplay_raw() -> bool:
        ffmpeg_bin = shutil.which("ffmpeg")
        aplay_bin = shutil.which("aplay")
        if not ffmpeg_bin or not aplay_bin:
            return False
        # Decode to raw 16-bit PCM. Using a fixed, common format tends to be robust.
        try:
            ff = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-v",
                    "error",
                    "-i",
                    str(wav_path),
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            ap = subprocess.Popen(
                [aplay_bin, "-q", "-f", "S16_LE", "-c", "2", "-r", "48000"],
                stdin=ff.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ff.stdout is not None:
                ff.stdout.close()
            ap_rc = ap.wait()
            ff_rc = ff.wait()
            if ap_rc == 0 and ff_rc == 0:
                time.sleep(0.10)
                return True
        except Exception:
            return False
        return False

    def _ffplay() -> bool:
        return _run_and_wait(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                "-loglevel",
                "error",
                "-sync",
                "audio",
                str(wav_path),
            ]
        )

    def _paplay() -> bool:
        return _run_and_wait(["paplay", str(wav_path)])

    def _pw_play() -> bool:
        # PipeWire native player on Ubuntu 22.04; usually the most reliable.
        # It may return slightly early on some setups, so we allow early-ok and rely on duration sleep.
        return _run_and_wait(["pw-play", str(wav_path)], allow_early_ok=True)

    def _aplay() -> bool:
        return _run_and_wait(["aplay", "-q", str(wav_path)])

    backend = backend.strip().lower()
    if backend != "auto":
        mapping = {
            "pw-play": _pw_play,
            "paplay": _paplay,
            "ffplay": _ffplay,
            "aplay": _aplay,
            "ffmpeg-aplay": _ffmpeg_aplay_raw,
        }
        fn = mapping.get(backend)
        if fn is None:
            print(f"Unknown backend: {backend}", file=sys.stderr)
            return False
        return fn()

    # Auto order (Ubuntu 22.04): PulseAudio (paplay) tends to be most reliable,
    # then PipeWire native (pw-play), then decode+ALSA, then ffplay.
    if _paplay():
        return True
    if _pw_play():
        return True
    if _ffmpeg_aplay_raw():
        return True
    if _ffplay():
        return True
    if _aplay():
        return True

    print("Could not play audio (tried pw-play, paplay, ffmpeg-aplay, ffplay, aplay)", file=sys.stderr)
    return False


def load_issues(tsv_path: Path, dataset_root: Path) -> List[Dict]:
    """Load issue list from TSV file."""
    issues = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Resolve relative path from TSV to absolute
            rel_wav = row['wav_path'].replace('../../../../wavs/', '')
            abs_wav = dataset_root / 'wavs' / rel_wav
            
            issues.append({
                'sim': float(row['sim']),
                'row': int(row['row']),
                'wav': row['wav'],
                'text': row['text'],
                'wav_path': abs_wav,
            })
    return issues


def verify_interactive(issues: List[Dict], output_path: Path, start_from: int = 0):
    """Interactive verification session."""
    results: List[Dict] = []
    total = len(issues)

    existing_by_idx: dict[int, Dict] = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    try:
                        idx = int(row.get("idx") or 0)
                    except Exception:
                        continue
                    if idx > 0:
                        existing_by_idx[idx] = row
            if existing_by_idx:
                print(f"💾 Найден прогресс: {len(existing_by_idx)} уже размечено в {output_path}")
        except Exception:
            existing_by_idx = {}
    
    print(f"\n{'='*80}")
    print(f"📋 Проверка {total} файлов с низким similarity")
    print(f"{'='*80}\n")
    print("Команды:")
    print("  [Enter]  - Прослушать снова")
    print("  ok       - Файл OK (данные правильные, проблема в Whisper)")
    print("  bad      - Файл BAD (плохое аудио или неверный текст)")
    print("  skip     - Пропустить (не уверен)")
    print("  quit     - Выйти и сохранить прогресс")
    print(f"{'='*80}\n")
    
    for idx, issue in enumerate(issues[start_from:], start=start_from + 1):
        if idx in existing_by_idx:
            # Already reviewed in a previous session; keep it.
            prev = existing_by_idx[idx]
            results.append(
                {
                    "idx": idx,
                    "sim": float(prev.get("sim") or issue["sim"]),
                    "row": int(prev.get("row") or issue["row"]),
                    "wav": prev.get("wav") or issue["wav"],
                    "text": prev.get("text") or issue["text"],
                    "verdict": (prev.get("verdict") or "skip").strip().lower(),
                }
            )
            continue
        wav_path = issue['wav_path']
        
        dur_s = None
        try:
            dur_s = _wav_duration_seconds(wav_path) if wav_path.exists() else None
        except Exception:
            dur_s = None

        print(f"\n{'─'*80}")
        if dur_s is None:
            print(f"📊 Файл {idx}/{total} | Similarity: {issue['sim']:.3f}")
        else:
            print(f"📊 Файл {idx}/{total} | Similarity: {issue['sim']:.3f} | Длительность: {dur_s:.2f}s")
        print(f"{'─'*80}")
        print(f"📄 Текст:     {issue['text']}")
        print(f"🎵 Аудио:     {issue['wav']}")
        print(f"📂 Путь:      {wav_path}")
        print(f"{'─'*80}")
        
        if not wav_path.exists():
            print(f"❌ ФАЙЛ НЕ НАЙДЕН: {wav_path}")
            verdict = 'missing'
        else:
            # Auto-play on first display
            play_audio(wav_path)
            
            while True:
                choice = input(f"\n[{idx}/{total}] Ваша оценка (ok/bad/skip/quit или Enter для повтора): ").strip().lower()
                
                if choice == '':
                    # Replay
                    play_audio(wav_path)
                    continue
                elif choice in ['ok', 'bad', 'skip', 'quit']:
                    verdict = choice
                    break
                else:
                    print("⚠️  Неизвестная команда. Используйте: ok, bad, skip, quit или Enter")
        
        if verdict == 'quit':
            print(f"\n💾 Сохранение прогресса... (проверено {idx-start_from}/{total})")
            break
        
        results.append({
            'idx': idx,
            'sim': issue['sim'],
            'row': issue['row'],
            'wav': issue['wav'],
            'text': issue['text'],
            'verdict': verdict,
        })
        
        # Auto-save every 5 files
        if len(results) % 5 == 0:
            _save_results(output_path, results)
            print(f"  💾 Автосохранение ({len(results)} проверено)")
    
    # Final save
    _save_results(output_path, results)
    
    print(f"\n{'='*80}")
    print(f"✅ Проверка завершена!")
    print(f"{'='*80}")
    _print_summary(results)
    print(f"\n📁 Результаты сохранены: {output_path}")


def play_list(issues: List[Dict], start_from: int = 0):
    """Play issues sequentially (no verdicts), pausing between files."""
    total = len(issues)

    print(f"\n{'='*80}")
    print(f"🎧 Прослушивание списка (без разметки): {total} файлов")
    print(f"{'='*80}\n")
    print("Команды:")
    print("  [Enter]  - Следующий файл")
    print("  r        - Повторить текущий")
    print("  q        - Выйти")
    print(f"{'='*80}\n")

    i = start_from
    while i < total:
        issue = issues[i]
        wav_path = issue["wav_path"]

        dur_s = None
        try:
            dur_s = _wav_duration_seconds(wav_path) if wav_path.exists() else None
        except Exception:
            dur_s = None

        print(f"\n{'─'*80}")
        if dur_s is None:
            print(f"📊 Файл {i+1}/{total} | Similarity: {issue['sim']:.3f}")
        else:
            print(f"📊 Файл {i+1}/{total} | Similarity: {issue['sim']:.3f} | Длительность: {dur_s:.2f}s")
        print(f"{'─'*80}")
        print(f"📄 Текст:     {issue['text']}")
        print(f"🎵 Аудио:     {issue['wav']}")
        print(f"📂 Путь:      {wav_path}")
        print(f"{'─'*80}")

        if not wav_path.exists():
            print(f"❌ ФАЙЛ НЕ НАЙДЕН: {wav_path}")
        else:
            play_audio(wav_path)

        choice = input("\n[Enter]=next, r=repeat, q=quit: ").strip().lower()
        if choice == "q":
            break
        if choice == "r":
            continue
        i += 1


def _save_results(output_path: Path, results: List[Dict]):
    """Save verification results to TSV."""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(results)


def _print_summary(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        print("Нет результатов")
        return
    
    ok_count = sum(1 for r in results if r['verdict'] == 'ok')
    bad_count = sum(1 for r in results if r['verdict'] == 'bad')
    skip_count = sum(1 for r in results if r['verdict'] == 'skip')
    missing_count = sum(1 for r in results if r['verdict'] == 'missing')
    
    print(f"  ✅ OK (ложная тревога):  {ok_count}")
    print(f"  ❌ BAD (реальная проблема): {bad_count}")
    print(f"  ⏭️  SKIP (пропущено):     {skip_count}")
    if missing_count > 0:
        print(f"  🔍 MISSING (не найдено):   {missing_count}")
    print(f"  📊 ВСЕГО проверено:       {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Interactive audio verification tool")
    parser.add_argument(
        'tsv_file',
        type=Path,
        help="TSV file with issues (e.g., low_similarity_list.tsv)"
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('../piper-training/datasets/felix_mirage_prepared_sr22050_seg15'),
        help="Dataset root directory (default: ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Output TSV file for results (default: <tsv_file>_verified.tsv)"
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help="Start from N-th file (0-indexed, default: 0)"
    )

    parser.add_argument(
        '--mode',
        choices=['verify', 'play'],
        default='verify',
        help='Mode: verify (mark ok/bad/skip) or play (just listen). Default: verify'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not load existing <output> progress; start fresh.'
    )

    parser.add_argument(
        '--player',
        type=str,
        default='auto',
        help='Playback backend: auto|pw-play|paplay|ffmpeg-aplay|ffplay|aplay (default: auto)'
    )

    parser.add_argument(
        '--diagnose-wav',
        type=Path,
        default=None,
        help='If set, run a quick diagnostics (duration + try players) for this wav and exit'
    )
    
    args = parser.parse_args()

    if args.diagnose_wav is not None:
        wav = args.diagnose_wav
        if not wav.exists():
            print(f"❌ WAV не найден: {wav}", file=sys.stderr)
            sys.exit(2)
        try:
            dur = _wav_duration_seconds(wav)
        except Exception as e:
            dur = None
        print(f"WAV: {wav}")
        print(f"wave_duration: {dur}")
        # ffprobe duration
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                out = subprocess.check_output(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(wav)],
                    text=True,
                ).strip()
                print(f"ffprobe_duration: {out}")
            except Exception as e:
                print(f"ffprobe_duration: error: {e}")
        # decode check
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                p = subprocess.run([ffmpeg, "-v", "error", "-i", str(wav), "-f", "null", "-"], capture_output=True, text=True)
                if p.stderr.strip():
                    print("ffmpeg_decode_errors:")
                    print(p.stderr.strip())
                else:
                    print("ffmpeg_decode_errors: none")
            except Exception as e:
                print(f"ffmpeg_decode_errors: error: {e}")

        for b in ["paplay", "pw-play", "ffmpeg-aplay", "ffplay", "aplay"]:
            ok = play_audio_with_backend(wav, backend=b)
            print(f"play_{b}: {ok}")
        sys.exit(0)
    
    if not args.tsv_file.exists():
        print(f"❌ Файл не найден: {args.tsv_file}", file=sys.stderr)
        sys.exit(1)
    
    if not args.dataset_root.exists():
        print(f"❌ Dataset root не найден: {args.dataset_root}", file=sys.stderr)
        sys.exit(1)
    
    output_path = args.output or args.tsv_file.parent / f"{args.tsv_file.stem}_verified.tsv"

    if args.no_resume and output_path.exists():
        output_path.unlink()
    
    issues = load_issues(args.tsv_file, args.dataset_root)
    print(f"📂 Загружено {len(issues)} файлов из {args.tsv_file}")
    
    # Store chosen player backend globally for the session.
    global play_audio
    chosen = args.player.strip().lower()
    if chosen != 'auto':
        def _play(wav_path: Path) -> bool:
            return play_audio_with_backend(wav_path, backend=chosen)
        play_audio = _play

    if args.mode == 'play':
        play_list(issues, args.start_from)
        return

    verify_interactive(issues, output_path, args.start_from)


if __name__ == '__main__':
    main()
