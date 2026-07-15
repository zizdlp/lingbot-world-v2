#!/usr/bin/env python3

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a LingBot World V2 generation job and download the video."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default="examples/03/image.jpg")
    parser.add_argument(
        "--action-path",
        default=None,
        metavar="DIR",
        help=(
            "Optional client-local trajectory directory containing poses.npy and "
            "intrinsics.npy; the server default is used when omitted."
        ),
    )
    parser.add_argument("--frame-num", type=int, default=361)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--request-id",
        default=None,
        help="Stable idempotency key; generated automatically when omitted.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        default=None,
        help="Exact client-local path for the downloaded video.",
    )
    output_group.add_argument(
        "--output-dir",
        default="output",
        help="Client-local download directory; the job id is used as the filename.",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Maximum wait in seconds; 0 waits indefinitely.",
    )
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.frame_num < 1:
        parser.error("--frame-num must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    return args


def json_request(method, url, payload=None, timeout=30):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", body)
        except json.JSONDecodeError:
            message = body
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to {url}: {exc.reason}") from exc


def submit_job(args):
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    payload = {
        "request_id": args.request_id,
        "prompt": args.prompt,
        "frame_num": args.frame_num,
        "seed": args.seed,
        "image_name": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
    }
    if args.action_path:
        action_path = Path(args.action_path)
        trajectory = {}
        for filename, field in (
            ("poses.npy", "poses_base64"),
            ("intrinsics.npy", "intrinsics_base64"),
        ):
            path = action_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"trajectory file not found: {path}")
            trajectory[field] = base64.b64encode(path.read_bytes()).decode("ascii")
        payload["trajectory"] = trajectory
    return json_request("POST", f"{args.server.rstrip('/')}/v1/jobs", payload)


def wait_for_job(args, job):
    status_url = urljoin(f"{args.server.rstrip('/')}/", f"v1/jobs/{job['id']}")
    deadline = time.monotonic() + args.timeout if args.timeout > 0 else None
    last_state = None
    while True:
        job = json_request("GET", status_url)
        position = job.get("queue_position")
        current_state = (job["status"], position)
        if current_state != last_state:
            queue_text = f", queue position {position}" if position else ""
            print(f"job {job['id']}: {job['status']}{queue_text}")
            last_state = current_state
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise RuntimeError(job.get("error") or "generation failed")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"job {job['id']} did not finish before timeout")
        time.sleep(args.poll_interval)


def download_video(args, job):
    video_url = urljoin(f"{args.server.rstrip('/')}/", job["video_url"].lstrip("/"))
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.output_dir) / f"{job['id']}.mp4"
    )
    partial_path = output_path.with_name(f"{output_path.name}.part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(video_url, headers={"Accept": "video/mp4"})
    try:
        with (
            urlopen(request, timeout=60) as response,
            partial_path.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(partial_path, output_path)
    except HTTPError as exc:
        raise RuntimeError(f"Video download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Video download failed: {exc.reason}") from exc
    finally:
        partial_path.unlink(missing_ok=True)
    print(f"video saved to {output_path}")


def main():
    args = parse_args()
    args.request_id = args.request_id or uuid.uuid4().hex
    try:
        print(f"request_id={args.request_id}")
        job = submit_job(args)
        print(
            f"submitted job {job['id']} (request_id={job['request_id']}, "
            f"trajectory={job.get('trajectory_source', 'default')})"
        )
        if args.no_wait:
            print(urljoin(f"{args.server.rstrip('/')}/", f"v1/jobs/{job['id']}"))
            return
        job = wait_for_job(args, job)
        download_video(args, job)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
