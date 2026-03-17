"""
Job worker that polls the Cloudflare Worker API, processes separation jobs,
and uploads/downloads results via the Worker R2 API. Supports multiple GPUs for parallel processing.
"""

import argparse
import contextlib
import logging
import os
import subprocess
import sys
import time
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_BASE = "https://runpod-gpu.zauen.workers.dev"
POLL_INTERVAL_SEC = 5

LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("job_worker")


def get_gpu_count() -> int:
    try:
        import torch
        count = max(1, torch.cuda.device_count())
        log.info("GPU count: %d", count)
        return count
    except Exception as e:
        log.warning("Could not detect GPUs (%s), using 1 worker", e)
        return 1


def claim_job(api_base: str) -> dict | None:
    log.debug("Claiming job from %s/job/claim", api_base)
    resp = requests.post(f"{api_base}/job/claim", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    job = data.get("job")
    if job:
        log.info("Claimed job id=%s file_url=%s", job.get("id"), job.get("file_url"))
    else:
        log.debug("No pending jobs")
    return job


def mark_done(api_base: str, job_id: str, result_url: str) -> None:
    log.info("[%s] Marking done result_url=%s", job_id, result_url)
    resp = requests.post(
        f"{api_base}/job/done",
        json={"id": job_id, "result_url": result_url},
        timeout=30,
    )
    resp.raise_for_status()
    log.debug("[%s] mark_done response status=%d", job_id, resp.status_code)


def mark_fail(api_base: str, job_id: str) -> None:
    log.warning("[%s] Marking failed", job_id)
    resp = requests.post(
        f"{api_base}/job/fail",
        json={"id": job_id},
        timeout=30,
    )
    resp.raise_for_status()
    log.debug("[%s] mark_fail response status=%d", job_id, resp.status_code)


def _resolve_download_url(file_url: str, api_base: str) -> str:
    """Resolve file_url to full download URL. file_url may be /download/{key} or absolute."""
    if file_url.startswith("http://") or file_url.startswith("https://"):
        return file_url
    base = api_base.rstrip("/")
    path = file_url if file_url.startswith("/") else f"/{file_url}"
    return f"{base}{path}"


def download_file(url: str, dest: Path) -> tuple[int, str]:
    """Download file, return (size_bytes, final_url)."""
    log.debug("Downloading %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "JobWorker/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        final_url = r.geturl()
    dest.write_bytes(data)
    size = len(data)
    log.info("Downloaded %d bytes to %s (final_url=%s)", size, dest, final_url)
    return size, final_url


def upload_to_worker(file_path: Path, api_base: str) -> str:
    """Upload file to Worker R2, return file_url (e.g. /download/{key}) for result_url."""
    size = file_path.stat().st_size
    log.debug("Uploading %s (%d bytes) to %s/upload", file_path, size, api_base)
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{api_base.rstrip('/')}/upload",
            files={"file": (file_path.name, f, "application/octet-stream")},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    file_url = data.get("file_url")
    if not file_url:
        raise RuntimeError(f"Upload response missing file_url: {data}")
    log.info("Uploaded to %s%s", api_base, file_url)
    return file_url


def _infer_extension_from_url(url: str) -> str | None:
    path = url.split("?")[0]
    ext = Path(path).suffix.lower()
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
        return ext
    return None


def _detect_audio_format(data: bytes) -> str:
    """Detect audio format from magic bytes. Fallback .wav for unknown."""
    if len(data) < 12:
        return ".wav"
    if data[:5] in (b"<!DOC", b"<html", b"<HTML") or data[:2] == b"<?":
        raise ValueError("Downloaded content appears to be HTML/XML, not audio")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return ".mp3"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".m4a"
    log.warning("Could not detect audio format from magic bytes, assuming .wav")
    return ".wav"


@contextlib.contextmanager
def _job_work_dir(prefix: str, keeptemp: bool, job_id: str):
    """Yield work directory. If keeptemp, do not delete (for debugging)."""
    if keeptemp:
        path = Path(tempfile.mkdtemp(prefix=f"{prefix}_{job_id}_"))
        try:
            yield path
        finally:
            log.info("[%s] Kept temp dir for inspection: %s", job_id, path)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
            yield Path(tmpdir)


def process_job(job: dict, gpu_id: int, api_base: str, gpu_count: int, keeptemp: bool = False) -> None:
    job_id = job["id"]
    file_url = job["file_url"]
    log.info("[%s] Processing job gpu_id=%s file_url=%s", job_id, gpu_id, file_url)

    with _job_work_dir("job_worker_", keeptemp, job_id) as tmp:
        output_dir = tmp / "separated"
        raw_path = tmp / "input_raw"

        try:
            download_url = _resolve_download_url(file_url, api_base)
            _, final_url = download_file(download_url, raw_path)
            ext = _infer_extension_from_url(final_url) or _infer_extension_from_url(file_url) or _detect_audio_format(raw_path.read_bytes())
            input_path = tmp / f"input_audio{ext}"
            raw_path.rename(input_path)
            log.debug("[%s] tmpdir=%s input_path=%s (detected ext=%s)", job_id, tmp, input_path, ext)
        except Exception as e:
            log.exception("[%s] Download failed: %s", job_id, e)
            mark_fail(api_base, job_id)
            return

        try:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            cmd = [
                sys.executable,
                "-m",
                "demucs_separator.run",
                str(input_path),
                "-o",
                str(output_dir),
            ]
            project_root = Path(__file__).resolve().parent
            log.info("[%s] Running separation: %s (cwd=%s)", job_id, " ".join(cmd), project_root)
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            output_lines: list[str] = []
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    log.info("[%s] %s", job_id, line)
            proc.wait()
            proc_output = "\n".join(output_lines)
            if proc.returncode != 0:
                log.error("[%s] demucs_separator exit=%d", job_id, proc.returncode)
                raise RuntimeError(
                    f"demucs_separator failed (exit {proc.returncode}): {proc_output}"
                )
            track_stem = input_path.stem
            vocals_htdemucs = output_dir / "htdemucs" / track_stem / "vocals.wav"
            vocals_mdx_extra = output_dir / "mdx_extra" / track_stem / "vocals.wav"
            if not vocals_htdemucs.exists():
                raise RuntimeError(f"Expected vocals at {vocals_htdemucs}")
            if not vocals_mdx_extra.exists():
                raise RuntimeError(f"Expected vocals at {vocals_mdx_extra}")
            log.info("[%s] Separation done: htdemucs=%s mdx_extra=%s", job_id, vocals_htdemucs, vocals_mdx_extra)
        except Exception as e:
            log.exception("[%s] Separation failed: %s", job_id, e)
            mark_fail(api_base, job_id)
            return

        try:
            zip_path = tmp / "vocals.zip"
            log.debug("[%s] Creating zip %s", job_id, zip_path)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(vocals_htdemucs, "vocals_htdemucs.wav")
                zf.write(vocals_mdx_extra, "vocals_mdx_extra.wav")
            zip_size = zip_path.stat().st_size
            log.info("[%s] Zip created %d bytes", job_id, zip_size)
            result_url = upload_to_worker(zip_path, api_base)
            mark_done(api_base, job_id, result_url)
            log.info("[%s] Job complete -> %s", job_id, result_url)
        except Exception as e:
            log.exception("[%s] Upload failed: %s", job_id, e)
            mark_fail(api_base, job_id)


def worker_loop(gpu_id: int, api_base: str, gpu_count: int, keeptemp: bool = False) -> None:
    log.info("[GPU %s] Worker started", gpu_id)
    while True:
        try:
            job = claim_job(api_base)
            if job:
                process_job(job, gpu_id, api_base, gpu_count, keeptemp)
            else:
                log.debug("[GPU %s] No job, sleeping %ds", gpu_id, POLL_INTERVAL_SEC)
                time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.exception("[GPU %s] Error: %s", gpu_id, e)
            time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Job worker for audio separation")
    parser.add_argument(
        "--keeptemp",
        action="store_true",
        help="Keep temp files after processing (for debugging)",
    )
    args = parser.parse_args()

    api_base = os.environ.get("JOB_API_BASE", API_BASE)
    gpu_count = get_gpu_count()
    workers = min(gpu_count, int(os.environ.get("WORKER_COUNT", gpu_count)))

    log.info("Starting job worker: workers=%d gpu_count=%d api=%s keeptemp=%s", workers, gpu_count, api_base, args.keeptemp)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(worker_loop, gpu_id, api_base, gpu_count, args.keeptemp)
            for gpu_id in range(workers)
        ]
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            log.info("Shutting down...")
            for f in futures:
                f.cancel()
            sys.exit(0)


if __name__ == "__main__":
    main()
