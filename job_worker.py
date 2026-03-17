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
TEMP_UPLOAD_BASE = "https://temp-host.zauen.workers.dev"
POLL_INTERVAL_SEC = 5

MIN_PART_SIZE = 5 * 1024 * 1024  # 5 MiB (required by R2 except for last part)
CHUNK_SIZE = 95 * 1024 * 1024   # 95 MiB = ~99.6 MB  ← leaves ~0.4 MB headroom for headers

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
    job = data.get("job") if isinstance(data, dict) else None
    if job and isinstance(job, dict) and job.get("id") and job.get("file_url"):
        log.info("Claimed job id=%s file_url=%s", job["id"], job["file_url"])
        return job
    if job is not None:
        log.warning("Claim response had job but invalid shape (missing id/file_url): %s", data)
    else:
        log.debug("No pending jobs (response: %s)", data)
    return None


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


def _safe_mark_fail(api_base: str, job_id: str, context: str = "") -> None:
    """Call mark_fail, log but do not raise if API call fails."""
    try:
        mark_fail(api_base, job_id)
    except Exception as e:
        log.exception("[%s] Failed to mark job as failed on API%s: %s", job_id, f" ({context})" if context else "", e)


def _resolve_download_url(file_url: str, file_base: str) -> str:
    """Resolve file_url to full download URL. All file transfers use file_base (temp-host), never api_base."""
    if file_url.startswith("http://") or file_url.startswith("https://"):
        return file_url
    base = file_base.rstrip("/")
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


def _mpu_create(upload_base: str, filename: str, content_type: str) -> dict:
    url = f"{upload_base.rstrip('/')}/uploads/{filename}"
    resp = requests.post(url, params={"action": "mpu-create", "contentType": content_type}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _mpu_upload_part(upload_base: str, filename: str, upload_id: str, part_number: int, chunk: bytes) -> dict:
    url = f"{upload_base.rstrip('/')}/uploads/{filename}"
    resp = requests.put(
        url,
        params={"action": "mpu-uploadpart", "uploadId": upload_id, "partNumber": part_number},
        data=chunk,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _mpu_complete(upload_base: str, filename: str, upload_id: str, parts: list[dict]) -> dict:
    url = f"{upload_base.rstrip('/')}/uploads/{filename}"
    resp = requests.post(
        url,
        params={"action": "mpu-complete", "uploadId": upload_id},
        json={"parts": parts},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _mpu_abort(upload_base: str, filename: str, upload_id: str) -> None:
    url = f"{upload_base.rstrip('/')}/uploads/{filename}"
    resp = requests.delete(url, params={"action": "mpu-abort", "uploadId": upload_id}, timeout=30)
    resp.raise_for_status()


def upload_file_multipart(file_path: Path, upload_base: str, content_type: str = "application/zip") -> str:
    """Upload file via R2 multipart API. Returns full download URL for result_url."""
    filename = file_path.name
    size = file_path.stat().st_size
    log.debug("Multipart upload %s (%d bytes) to %s", file_path, size, upload_base)

    session = _mpu_create(upload_base, filename, content_type)
    upload_id = session["uploadId"]
    log.debug("Multipart session created uploadId=%s", upload_id)

    parts: list[dict] = []
    try:
        with open(file_path, "rb") as f:
            part_number = 1
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                is_last = len(chunk) < CHUNK_SIZE
                if not is_last and len(chunk) < MIN_PART_SIZE:
                    raise ValueError(
                        f"Part {part_number} is {len(chunk)} bytes; "
                        f"all non-final parts must be >= {MIN_PART_SIZE} bytes"
                    )
                result = _mpu_upload_part(upload_base, filename, upload_id, part_number, chunk)
                parts.append({"partNumber": result["partNumber"], "etag": result["etag"]})
                log.debug("Uploaded part %d (%d bytes)", part_number, len(chunk))
                part_number += 1

        result = _mpu_complete(upload_base, filename, upload_id, parts)
        url = result.get("url")
        if not url:
            raise RuntimeError(f"Multipart complete response missing url: {result}")
        log.info("Multipart upload complete -> %s", url)
        return url
    except Exception as e:
        log.warning("Multipart upload failed, aborting: %s", e)
        _mpu_abort(upload_base, filename, upload_id)
        raise


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
    file_base = os.environ.get("TEMP_UPLOAD_BASE", TEMP_UPLOAD_BASE)
    log.info("[%s] Processing job gpu_id=%s file_url=%s", job_id, gpu_id, file_url)

    with _job_work_dir("job_worker_", keeptemp, job_id) as tmp:
        output_dir = tmp / "separated"
        raw_path = tmp / "input_raw"

        try:
            download_url = _resolve_download_url(file_url, file_base)
            _, final_url = download_file(download_url, raw_path)
            ext = _infer_extension_from_url(final_url) or _infer_extension_from_url(file_url) or _detect_audio_format(raw_path.read_bytes())
            input_path = tmp / f"input_audio{ext}"
            raw_path.rename(input_path)
            log.debug("[%s] tmpdir=%s input_path=%s (detected ext=%s)", job_id, tmp, input_path, ext)
        except Exception as e:
            log.exception("[%s] Download failed: %s", job_id, e)
            _safe_mark_fail(api_base, job_id, "download")
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
            _safe_mark_fail(api_base, job_id, "separation")
            return

        try:
            zip_path = tmp / f"vocals_{job_id}.zip"
            log.debug("[%s] Creating zip %s", job_id, zip_path)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                zf.write(vocals_htdemucs, "vocals_htdemucs.wav")
                zf.write(vocals_mdx_extra, "vocals_mdx_extra.wav")
            zip_size = zip_path.stat().st_size
            log.info("[%s] Zip created %d bytes", job_id, zip_size)
            upload_base = os.environ.get("TEMP_UPLOAD_BASE", TEMP_UPLOAD_BASE)
            result_url = upload_file_multipart(zip_path, upload_base, content_type="application/zip")
            mark_done(api_base, job_id, result_url)
            log.info("[%s] Job complete -> %s", job_id, result_url)
        except Exception as e:
            log.exception("[%s] Upload failed: %s", job_id, e)
            _safe_mark_fail(api_base, job_id, "upload")


def worker_loop(gpu_id: int, api_base: str, gpu_count: int, keeptemp: bool = False) -> None:
    log.info("[GPU %s] Worker started", gpu_id)
    while True:
        try:
            job = claim_job(api_base)
            if job:
                try:
                    process_job(job, gpu_id, api_base, gpu_count, keeptemp)
                except Exception as e:
                    log.exception("[GPU %s] Job failed (marking as failed on API): %s", gpu_id, e)
                    _safe_mark_fail(api_base, job["id"], "worker_loop")
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

    upload_base = os.environ.get("TEMP_UPLOAD_BASE", TEMP_UPLOAD_BASE)
    log.info(
        "Starting job worker: workers=%d gpu_count=%d api=%s upload=%s keeptemp=%s",
        workers, gpu_count, api_base, upload_base, args.keeptemp,
    )

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
