"""
Fix voice recorder WAV files for Windows compatibility.

Many mini voice recorders produce WAV files in non-standard formats
(8-bit, ADPCM, unusual sample rates, bad headers) that Windows Media Player
and some apps cannot play. This script converts them to standard PCM 16-bit WAV,
which plays everywhere on Windows.

Usage:
  python fix_wav.py <file.wav> [<file2.wav> ...]
  python fix_wav.py <directory>   # processes all .wav in directory

Output: writes fixed files as <name>_fixed.wav in the same folder.
        Use --replace to overwrite originals instead.

If soundfile fails (rare formats), install ffmpeg and use:
  python compress_audio.py <file.wav> --format wav16
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf


def _fix_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path), "-acodec", "pcm_s16le", "-y", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ffmpeg ERROR: {result.stderr.strip() or 'unknown'}")
        return False
    return True


def fix_wav(input_path: Path, output_path: Path) -> bool:
    """Convert WAV to standard PCM 16-bit. Returns True on success."""
    try:
        data, sample_rate = sf.read(str(input_path), dtype="int16")
        sf.write(str(output_path), data, sample_rate, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"  soundfile failed: {e}")
        if output_path.exists():
            output_path.unlink()
        if shutil.which("ffmpeg"):
            print("  Trying ffmpeg fallback...")
            return _fix_with_ffmpeg(input_path, output_path)
        print("  Install ffmpeg and run: python compress_audio.py <file> --format wav16")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix voice recorder WAV files for Windows playback.",
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="WAV file(s) or directory to process.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite originals instead of creating *_fixed.wav",
    )
    args = parser.parse_args()

    wav_files: list[Path] = []
    for p in args.paths:
        if not p.exists():
            print(f"Error: '{p}' does not exist.")
            sys.exit(1)
        if p.is_file():
            if p.suffix.lower() == ".wav":
                wav_files.append(p)
            else:
                print(f"Skipping non-WAV: {p}")
        else:
            wav_files.extend(sorted(p.rglob("*.wav")))

    if not wav_files:
        print("No WAV files found.")
        sys.exit(0)

    print(f"Fixing {len(wav_files)} WAV file(s)...\n")
    ok = 0
    for wav in wav_files:
        if args.replace:
            out = wav.parent / (wav.stem + "_tmp.wav")
            success = fix_wav(wav, out)
            if success:
                out.replace(wav)
                ok += 1
                print(f"  OK: {wav.name}")
            else:
                out.unlink(missing_ok=True)
        else:
            out = wav.parent / (wav.stem + "_fixed.wav")
            success = fix_wav(wav, out)
            if success:
                ok += 1
                print(f"  OK: {wav.name} -> {out.name}")
    print(f"\nDone. Fixed {ok}/{len(wav_files)} files.")


if __name__ == "__main__":
    main()
