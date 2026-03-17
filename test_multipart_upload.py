"""
Test script for multipart upload to Cloudflare R2 Worker API.
Usage: python test_multipart_upload.py <file_path> [--content-type <mime>]
"""

import argparse
import mimetypes
import sys
from pathlib import Path

import requests

BASE_URL = "https://temp-host.zauen.workers.dev"
MIN_PART_SIZE = 5 * 1024 * 1024  # 5 MiB (required by R2 except for last part)
CHUNK_SIZE = 95 * 1024 * 1024   # 95 MiB = ~99.6 MB  ← leaves ~0.4 MB headroom for headers


def get_content_type(file_path: Path, user_content_type: str | None) -> str:
    if user_content_type:
        return user_content_type
    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed or "application/octet-stream"


def create_upload_session(filename: str, content_type: str) -> dict:
    url = f"{BASE_URL}/uploads/{filename}"
    resp = requests.post(url, params={"action": "mpu-create", "contentType": content_type})
    resp.raise_for_status()
    return resp.json()


def upload_part(filename: str, upload_id: str, part_number: int, chunk: bytes) -> dict:
    url = f"{BASE_URL}/uploads/{filename}"
    resp = requests.put(
        url,
        params={"action": "mpu-uploadpart", "uploadId": upload_id, "partNumber": part_number},
        data=chunk,
    )
    resp.raise_for_status()
    return resp.json()


def complete_upload(filename: str, upload_id: str, parts: list[dict]) -> dict:
    url = f"{BASE_URL}/uploads/{filename}"
    resp = requests.post(
        url,
        params={"action": "mpu-complete", "uploadId": upload_id},
        json={"parts": parts},
    )
    resp.raise_for_status()
    return resp.json()


def abort_upload(filename: str, upload_id: str) -> None:
    url = f"{BASE_URL}/uploads/{filename}"
    resp = requests.delete(url, params={"action": "mpu-abort", "uploadId": upload_id})
    resp.raise_for_status()


def upload_file(file_path: Path, content_type: str | None = None) -> dict:
    content_type = get_content_type(file_path, content_type)
    filename = file_path.name
    key = f"uploads/{filename}"

    file_size = file_path.stat().st_size

    # Step 1: Create multipart session
    print(f"Creating upload session for {filename} ({file_size:,} bytes)...")
    session = create_upload_session(filename, content_type)
    upload_id = session["uploadId"]
    print(f"  uploadId: {upload_id}")

    parts: list[dict] = []
    try:
        # Step 2: Upload parts
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

                print(f"  Uploading part {part_number} ({len(chunk):,} bytes)...")
                result = upload_part(filename, upload_id, part_number, chunk)
                parts.append({"partNumber": result["partNumber"], "etag": result["etag"]})
                part_number += 1

        # Step 3: Complete upload
        print("Completing upload...")
        result = complete_upload(filename, upload_id, parts)
        return result

    except Exception as e:
        print(f"Error: {e}. Aborting upload...")
        abort_upload(filename, upload_id)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Multipart upload test for R2 Worker API")
    parser.add_argument("file_path", type=Path, help="Path to file to upload")
    parser.add_argument(
        "--content-type",
        "-t",
        help="MIME type (e.g. video/mp4, image/jpeg). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        return 1

    if not args.file_path.is_file():
        print(f"Error: Not a file: {args.file_path}", file=sys.stderr)
        return 1

    try:
        result = upload_file(args.file_path, args.content_type)
        print("\nUpload successful!")
        print(f"  key: {result['key']}")
        print(f"  url: {result['url']}")
        return 0
    except requests.RequestException as e:
        print(f"\nRequest failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            try:
                err_body = e.response.json()
                print(f"  Response: {err_body}", file=sys.stderr)
            except Exception:
                print(f"  Response text: {e.response.text[:500]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
