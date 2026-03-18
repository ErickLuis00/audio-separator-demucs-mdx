"""
Audio compressor for large WAV files produced by Demucs/MDX.

Supports:
  - WAV16 : stay WAV, convert 32-bit float → 16-bit PCM. ~50% smaller, CD quality (recommended)
  - FLAC  : lossless, ~50% smaller, keeps full bit depth
  - OPUS  : lossy, ~90% smaller, 128 kbps — fastest encode, transparent quality (recommended lossy)
  - MP3   : lossy, ~90% smaller, 128 kbps (matches voice/compressed source quality)
  - OGG   : lossy, ~85% smaller, quality 9 (~320 kbps equivalent)

Why demucs outputs double the size:
  Demucs writes 32-bit float WAV internally. If your input was 16-bit PCM (standard),
  the stems will be exactly 2x larger. Converting back to 16-bit PCM halves the size
  with no perceptible quality loss — 16-bit is CD quality (96 dB dynamic range).

Performance:
  All conversions use ffmpeg directly (C/SIMD, much faster than pure Python).
  Multiple files are processed in parallel using threads (each file = one ffmpeg process).

Usage:
  python compress_audio.py <input_dir_or_file> [--format wav16|flac|opus|mp3|ogg]
                           [--workers N] [--delete-originals] [--output-dir DIR]

Dependencies:
  ffmpeg must be installed and on PATH.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

_print_lock = Lock()


def _get_duration_sec(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _parse_time(s: str) -> float | None:
    m = re.search(r"time=(\d+):(\d+):(\d+)\.(\d+)", s)
    if not m:
        return None
    h, m1, s, frac = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    return h * 3600 + m1 * 60 + s + int(frac) / (10 ** len(frac))


def _ffmpeg_progress(args: list[str], input_path: Path, label: str, lock: Lock | None) -> None:
    duration = _get_duration_sec(input_path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-stats", *args]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)
    last_pct = -1
    while True:
        line = proc.stderr.readline()
        if not line and proc.poll() is not None:
            break
        if duration and duration > 0:
            t = _parse_time(line)
            if t is not None:
                pct = min(99, int(100 * t / duration))
                if pct != last_pct and pct % 5 == 0:
                    last_pct = pct
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    out = f"\r  {label[:40]:<40} [{bar}] {pct:>3}%\r"
                    if lock:
                        with lock:
                            sys.stderr.write(out)
                            sys.stderr.flush()
                    else:
                        sys.stderr.write(out)
                        sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    if duration and last_pct >= 0:
        clear = "\r" + " " * 70 + "\r"
        if lock:
            with lock:
                sys.stderr.write(clear)
                sys.stderr.flush()
        else:
            sys.stderr.write(clear)
            sys.stderr.flush()


def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _ffmpeg(*args: str) -> None:
    """Run ffmpeg silently, raise RuntimeError on failure."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def _run_ffmpeg(args: list[str], wav_path: Path, show_progress: bool, lock: Lock | None) -> None:
    if show_progress:
        _ffmpeg_progress(args, wav_path, wav_path.name, lock)
    else:
        _ffmpeg(*args)


def convert_to_wav16(wav_path: Path, output_dir: Path, show_progress: bool = False, lock: Lock | None = None) -> Path:
    out_path = output_dir / wav_path.name
    in_place = out_path.resolve() == wav_path.resolve()
    args = ["-i", str(wav_path), "-acodec", "pcm_s16le", "-y"]

    if in_place:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=output_dir) as tmp:
            tmp_path = Path(tmp.name)
        args.append(str(tmp_path))
        try:
            _run_ffmpeg(args, wav_path, show_progress, lock)
            shutil.move(str(tmp_path), str(out_path))
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        args.append(str(out_path))
        _run_ffmpeg(args, wav_path, show_progress, lock)
    return out_path


def convert_to_flac(wav_path: Path, output_dir: Path, show_progress: bool = False, lock: Lock | None = None) -> Path:
    out_path = output_dir / wav_path.with_suffix(".flac").name
    args = ["-i", str(wav_path), "-acodec", "flac", "-compression_level", "8", "-y", str(out_path)]
    _run_ffmpeg(args, wav_path, show_progress, lock)
    return out_path


def convert_to_opus(wav_path: Path, output_dir: Path, show_progress: bool = False, lock: Lock | None = None) -> Path:
    out_path = output_dir / wav_path.with_suffix(".opus").name
    # Opus: fastest lossy encoder, 128k = transparent, widely supported (Chrome, Firefox, Android)
    args = ["-i", str(wav_path), "-codec:a", "libopus", "-b:a", "128k", "-y", str(out_path)]
    _run_ffmpeg(args, wav_path, show_progress, lock)
    return out_path


def convert_to_mp3(wav_path: Path, output_dir: Path, show_progress: bool = False, lock: Lock | None = None) -> Path:
    out_path = output_dir / wav_path.with_suffix(".mp3").name
    # 128k: output bitrate cannot exceed source quality; voice/IMA ADPCM (~96k equiv) gains nothing from 256k
    args = ["-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k", "-y", str(out_path)]
    _run_ffmpeg(args, wav_path, show_progress, lock)
    return out_path


def convert_to_ogg(wav_path: Path, output_dir: Path, show_progress: bool = False, lock: Lock | None = None) -> Path:
    out_path = output_dir / wav_path.with_suffix(".ogg").name
    args = ["-i", str(wav_path), "-codec:a", "libvorbis", "-q:a", "9", "-y", str(out_path)]
    _run_ffmpeg(args, wav_path, show_progress, lock)
    return out_path


CONVERTERS = {
    "wav16": convert_to_wav16,
    "flac": convert_to_flac,
    "opus": convert_to_opus,
    "mp3": convert_to_mp3,
    "ogg": convert_to_ogg,
}


def _collect_wav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            print(f"Error: '{input_path}' is not a WAV file.")
            sys.exit(1)
        return [input_path]
    return sorted(input_path.rglob("*.wav"))


def _convert_one(
    wav_path: Path,
    fmt: str,
    output_dir: Path | None,
    delete_originals: bool,
    show_progress: bool = False,
    lock: Lock | None = None,
) -> tuple[Path, float, int, int, str | None]:
    """Convert a single file. Returns (wav_path, elapsed, orig_size, out_size, error)."""
    dest = output_dir or wav_path.parent
    dest.mkdir(parents=True, exist_ok=True)

    original_size = wav_path.stat().st_size
    t0 = time.perf_counter()

    try:
        out_path = CONVERTERS[fmt](wav_path, dest, show_progress, lock)
        elapsed = time.perf_counter() - t0
        compressed_size = out_path.stat().st_size

        if delete_originals and out_path.resolve() != wav_path.resolve():
            wav_path.unlink()

        return wav_path, elapsed, original_size, compressed_size, None
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return wav_path, elapsed, original_size, 0, str(exc)


def _process_files(
    wav_files: list[Path],
    fmt: str,
    delete_originals: bool,
    output_dir: Path | None,
    workers: int,
) -> None:
    if fmt == "wav16" and output_dir is None:
        print("  [wav16] Files will be overwritten in-place (32-bit → 16-bit PCM).")

    label = "WAV 16-bit PCM" if fmt == "wav16" else fmt.upper()
    effective_workers = min(workers, len(wav_files))
    parallel_note = f"  ({effective_workers} worker{'s' if effective_workers > 1 else ''} in parallel)"
    print(f"\nConverting {len(wav_files)} WAV file(s) → {label}{parallel_note}\n")
    print(f"  {'File':<45} {'Original':>10}  {'Output':>10}  {'Saved':>7}  {'Time':>8}")
    print("  " + "-" * 90)

    total_original = 0
    total_compressed = 0
    total_start = time.perf_counter()

    # Results arrive out of order when parallel — collect then print in original order
    results: dict[Path, tuple[float, int, int, str | None]] = {}

    show_progress = effective_workers == 1
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(_convert_one, f, fmt, output_dir, delete_originals, show_progress, _print_lock if show_progress else None): f
            for f in wav_files
        }
        for future in as_completed(futures):
            wav_path, elapsed, orig_size, out_size, error = future.result()
            results[wav_path] = (elapsed, orig_size, out_size, error)
            # Print as each file finishes so the user sees live progress
            with _print_lock:
                if error:
                    print(f"  ERROR {wav_path.name}: {error}")
                else:
                    saved_pct = (1 - out_size / orig_size) * 100
                    in_place = (output_dir is None or (output_dir / wav_path.name).resolve() == wav_path.resolve())
                    note = " [in-place]" if fmt == "wav16" and in_place else ""
                    print(
                        f"  {wav_path.name:<45}"
                        f" {_sizeof_fmt(orig_size):>10}"
                        f"  {_sizeof_fmt(out_size):>10}"
                        f"  {saved_pct:>6.1f}%"
                        f"  {elapsed:>6.1f}s"
                        f"{note}"
                    )

    for _, (_, orig_size, out_size, error) in results.items():
        if not error:
            total_original += orig_size
            total_compressed += out_size

    total_elapsed = time.perf_counter() - total_start
    print("  " + "-" * 90)
    total_saved_pct = (1 - total_compressed / total_original) * 100 if total_original else 0
    print(
        f"  {'TOTAL':<45}"
        f" {_sizeof_fmt(total_original):>10}"
        f"  {_sizeof_fmt(total_compressed):>10}"
        f"  {total_saved_pct:>6.1f}%"
        f"  {total_elapsed:>6.1f}s  (wall clock)"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress large WAV files produced by Demucs/MDX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A single WAV file or a directory of WAV files (searched recursively).",
    )
    parser.add_argument(
        "--format",
        choices=["wav16", "flac", "opus", "mp3", "ogg"],
        default="wav16",
        help=(
            "Output format. "
            "'wav16' = stay WAV, 32-bit→16-bit PCM, ~50%% smaller, no quality loss (default). "
            "'flac' = lossless, ~50%% smaller, keeps full bit depth. "
            "'opus' = lossy 128 kbps .opus, fastest encode, transparent quality (recommended lossy). "
            "'mp3' = lossy 128 kbps ~90%% smaller. "
            "'ogg' = lossy quality-9 ~85%% smaller."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of files to convert in parallel (default: 4). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write compressed files. Defaults to same directory as source WAVs.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Remove source WAV files after successful compression.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: '{args.input}' does not exist.")
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH. Please install ffmpeg.")
        sys.exit(1)

    wav_files = _collect_wav_files(args.input)
    if not wav_files:
        print(f"No WAV files found in: {args.input}")
        sys.exit(0)

    _process_files(
        wav_files=wav_files,
        fmt=args.format,
        delete_originals=args.delete_originals,
        output_dir=args.output_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
