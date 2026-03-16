"""
Local vocal separation using demucs-infer (htdemucs + mdx_extra).
Runs on your PC, no cloud/GPU service.
"""

import shutil
import subprocess
import sys
from pathlib import Path

MODELS = ("htdemucs", "mdx_extra")


def _get_demucs_infer_cmd() -> list[str]:
    exe = shutil.which("demucs-infer")
    if exe:
        return [exe]
    script_dir = Path(sys.executable).parent
    for name in ("demucs-infer.exe", "demucs-infer"):
        candidate = script_dir / name
        if candidate.exists():
            return [str(candidate)]
    return [sys.executable, "-m", "demucs_infer"]


def separate_vocals_local(
    input_path: Path,
    output_dir: Path | None = None,
    models: tuple[str, ...] = MODELS,
) -> dict[str, Path]:
    """
    Run demucs-infer vocal separation locally.
    Returns dict of model_name -> vocals.wav path.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = output_dir or (input_path.parent / "separated")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}

    for model in models:
        model_out = out_dir / model
        track_name = input_path.stem
        vocals_path = model_out / track_name / "vocals.wav"

        args = _get_demucs_infer_cmd() + [
            "-n",
            model,
            "--two-stems=vocals",
            "-o",
            str(out_dir),
            str(input_path),
        ]

        proc = subprocess.run(args)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs-infer failed for {model} (exit code {proc.returncode})")

        if not vocals_path.exists():
            raise RuntimeError(f"Expected vocals at {vocals_path}")

        result[model] = vocals_path

    return result
