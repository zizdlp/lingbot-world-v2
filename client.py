#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SUPPORTED_SIZES = ("720*1280", "1280*720", "480*832", "832*480")
ARRAY_INPUTS = (
    ("poses.npy", "poses", "poses_base64"),
    ("intrinsics.npy", "intrinsics", "intrinsics_base64"),
    ("action.npy", "action", "action_base64"),
    ("wasd_action.npy", "wasd_action", "wasd_action_base64"),
    ("ijkl_action.npy", "ijkl_action", "ijkl_action_base64"),
)


def parse_timesteps(value):
    try:
        timesteps = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timesteps must be comma-separated integers"
        ) from exc
    if not 1 <= len(timesteps) <= 16:
        raise argparse.ArgumentTypeError("provide between 1 and 16 timesteps")
    if any(timestep < 0 or timestep >= 1000 for timestep in timesteps):
        raise argparse.ArgumentTypeError("timestep indices must be between 0 and 999")
    if any(left >= right for left, right in zip(timesteps, timesteps[1:])):
        raise argparse.ArgumentTypeError("timestep indices must be strictly increasing")
    return timesteps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a LingBot World V2 generation job and download the video."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--task", default="i2v-A14B")
    parser.add_argument(
        "--input-dir",
        default=None,
        metavar="DIR",
        help="Optional directory containing prompt.txt, image.jpg, and input arrays.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", default=None)
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image override; the server default is used when omitted.",
    )
    parser.add_argument(
        "--action-path",
        default=None,
        metavar="DIR",
        help=(
            "Optional client-local directory containing trajectory/action arrays; "
            "individual file arguments take precedence."
        ),
    )
    parser.add_argument("--poses", default=None, help="Path to poses.npy.")
    parser.add_argument("--intrinsics", default=None, help="Path to intrinsics.npy.")
    parser.add_argument("--action", default=None, help="Path to action.npy.")
    parser.add_argument(
        "--wasd-action", default=None, help="Path to wasd_action.npy."
    )
    parser.add_argument(
        "--ijkl-action", default=None, help="Path to ijkl_action.npy."
    )
    parser.add_argument("--size", choices=SUPPORTED_SIZES, default=None)
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument(
        "--timesteps-index",
        type=parse_timesteps,
        default=None,
        metavar="I0,I1,...",
    )
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
    parser.add_argument(
        "--show-capabilities",
        action="store_true",
        help="Print server-owned parameters and request defaults, then exit.",
    )
    args = parser.parse_args()
    if args.frame_num is not None and args.frame_num < 1:
        parser.error("--frame-num must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.sample_shift is not None and (
        not math.isfinite(args.sample_shift) or args.sample_shift <= 0
    ):
        parser.error("--sample-shift must be a positive finite number")
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


def require_directory(value, option):
    if value is None:
        return None
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{option} directory not found: {path}")
    return path


def optional_file(value, fallback_dirs, filename, label):
    if value is not None:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    for directory in fallback_dirs:
        if directory is not None and (directory / filename).is_file():
            return directory / filename
    return None


def optional_image(value, input_dir):
    if value is not None:
        return optional_file(value, (), "image.jpg", "image")
    if input_dir is not None:
        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            path = input_dir / f"image{suffix}"
            if path.is_file():
                return path
    return None


def submit_job(args):
    input_dir = require_directory(args.input_dir, "--input-dir")
    action_path = require_directory(args.action_path, "--action-path")
    payload = {
        "request_id": args.request_id,
        "task": args.task,
    }

    prompt = args.prompt
    prompt_path = None
    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    elif prompt is None and input_dir is not None:
        candidate = input_dir / "prompt.txt"
        if candidate.is_file():
            prompt_path = candidate
    if prompt_path is not None:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"prompt file is empty: {prompt_path}")
    if prompt is not None:
        payload["prompt"] = prompt

    image_path = optional_image(args.image, input_dir)
    if image_path is not None:
        payload["image_name"] = image_path.name
        payload["image_base64"] = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

    trajectory = {}
    for filename, argument, field in ARRAY_INPUTS:
        path = optional_file(
            getattr(args, argument),
            (action_path, input_dir),
            filename,
            filename,
        )
        if path is not None:
            trajectory[field] = base64.b64encode(path.read_bytes()).decode("ascii")
    if trajectory:
        payload["trajectory"] = trajectory

    for field in (
        "size",
        "frame_num",
        "seed",
        "sample_shift",
        "timesteps_index",
    ):
        value = getattr(args, field)
        if value is not None:
            payload[field] = value
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
    if args.show_capabilities:
        try:
            capabilities = json_request(
                "GET", f"{args.server.rstrip('/')}/v1/capabilities"
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(json.dumps(capabilities, ensure_ascii=False, indent=2))
        return
    args.request_id = args.request_id or uuid.uuid4().hex
    try:
        print(f"request_id={args.request_id}")
        job = submit_job(args)
        print(
            f"submitted job {job['id']} (request_id={job['request_id']}, "
            f"inputs={job.get('input_sources', {})})"
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
