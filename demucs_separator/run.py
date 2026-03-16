"""
CLI for local demucs-infer vocal separation (htdemucs + mdx_extra).

Usage:
  python -m demucs_separator.run bolsoremix.wav
  python -m demucs_separator.run --model mdx_extra bolsoremix.wav
  python -m demucs_separator.run --model htdemucs bolsoremix.wav
"""

import argparse
import sys
from pathlib import Path

from demucs_separator.app import MODELS, separate_vocals_local


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local vocal separation via demucs-infer (htdemucs + mdx_extra)"
    )
    parser.add_argument(
        "tracks",
        nargs="+",
        help="Audio file(s) to separate (vocals only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/separated)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODELS),
        default=None,
        help="Single model to run (default: both htdemucs and mdx_extra)",
    )
    args = parser.parse_args()

    models = (args.model,) if args.model else MODELS

    for path_str in args.tracks:
        path = Path(path_str).resolve()
        if not path.exists():
            print(f"File not found: {path}", flush=True)
            continue

        out_dir = args.output or (path.parent / "separated")
        print(f"Processing: {path} (models: {', '.join(models)})", flush=True)

        try:
            result = separate_vocals_local(path, output_dir=out_dir, models=models)
            for model, vocals_path in result.items():
                print(f"  -> {model}: {vocals_path}", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            sys.exit(1)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
