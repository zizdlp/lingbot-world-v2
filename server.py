#!/usr/bin/env python3

import argparse
import base64
import binascii
import gc
import io
import json
import logging
import math
import os
import queue
import re
import shutil
import sqlite3
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, UnidentifiedImageError


SUPPORTED_SIZES = ("720*1280", "1280*720", "480*832", "832*480")
SUPPORTED_TASKS = ("i2v-A14B",)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DEFAULT_PROMPT = "A serene lakeside scene with a lone tree standing in calm water."
DEFAULT_TIMESTEPS_INDEX = (0, 250, 500, 750)
ARRAY_INPUTS = (
    ("poses.npy", "poses_base64", (4, 4), True),
    ("intrinsics.npy", "intrinsics_base64", (4,), True),
    ("action.npy", "action_base64", (4,), False),
    ("wasd_action.npy", "wasd_action_base64", (4,), False),
    ("ijkl_action.npy", "ijkl_action_base64", (4,), False),
)


class QueueCapacityError(Exception):
    pass


def parse_timesteps(value):
    try:
        timesteps = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timesteps must be comma-separated integers"
        ) from exc
    if not 1 <= len(timesteps) <= 16:
        raise argparse.ArgumentTypeError("provide between 1 and 16 timesteps")
    if any(index < 0 or index >= 1000 for index in timesteps):
        raise argparse.ArgumentTypeError("timestep indices must be between 0 and 999")
    if any(left >= right for left, right in zip(timesteps, timesteps[1:])):
        raise argparse.ArgumentTypeError("timestep indices must be strictly increasing")
    return timesteps


def default_prompt(args):
    if args.default_prompt is not None:
        return args.default_prompt.strip()
    prompt_path = Path(args.action_path) / "prompt.txt"
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def default_image_path(args):
    if args.default_image is not None:
        return Path(args.default_image)
    input_dir = Path(args.action_path)
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = input_dir / f"image{suffix}"
        if path.is_file():
            return path
    return input_dir / "image.jpg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persistent 8-GPU HTTP inference service for LingBot World V2."
    )
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="i2v-A14B")
    parser.add_argument("--ckpt-dir", default="lingbot-world-v2-14b-causal-fast")
    parser.add_argument(
        "--action-path",
        default="examples/03",
        help="Default input directory containing image and trajectory/action arrays.",
    )
    parser.add_argument("--default-image", default=None)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="/mnt/data/lingbot-world-v2")
    parser.add_argument(
        "--output-dir",
        default="/mnt/outputs/lingbot-world-v2",
        help="Directory for generated videos; keep this on persistent storage.",
    )
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--retention-hours", type=int, default=168)
    parser.add_argument("--size", choices=SUPPORTED_SIZES, default="480*832")
    parser.add_argument("--frame-num", type=int, default=361)
    parser.add_argument("--max-frame-num", type=int, default=361)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--ulysses-size", type=int, default=8)
    parser.add_argument(
        "--dit-fsdp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--t5-fsdp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--sample-shift", type=float, default=10.0)
    parser.add_argument(
        "--timesteps-index",
        type=parse_timesteps,
        default=DEFAULT_TIMESTEPS_INDEX,
        metavar="I0,I1,...",
    )
    parser.add_argument("--max-attention-size", type=int, default=None)
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=32,
        help="Maximum size of each uploaded image or trajectory file.",
    )
    args = parser.parse_args()

    if args.frame_num < 1 or args.max_frame_num < 1:
        parser.error("frame counts must be positive")
    if args.frame_num > args.max_frame_num:
        parser.error("--frame-num cannot exceed --max-frame-num")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.local_attn_size != -1 and args.local_attn_size < args.chunk_size:
        parser.error("--local-attn-size must be -1 or at least --chunk-size")
    if args.sink_size < 0:
        parser.error("--sink-size cannot be negative")
    if args.local_attn_size != -1 and args.sink_size >= args.local_attn_size:
        parser.error("--sink-size must be smaller than --local-attn-size")
    if args.max_queue_size < 1:
        parser.error("--max-queue-size must be positive")
    if args.retention_hours < 0:
        parser.error("--retention-hours cannot be negative")
    if args.max_upload_mb < 1:
        parser.error("--max-upload-mb must be positive")
    if args.max_attention_size is not None and args.max_attention_size < 1:
        parser.error("--max-attention-size must be positive")
    if not math.isfinite(args.sample_shift) or args.sample_shift <= 0:
        parser.error("--sample-shift must be a positive finite number")
    if not default_prompt(args):
        parser.error("the default prompt must not be empty")
    return args


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ServiceState:
    def __init__(self, args, world_size):
        self.args = args
        self.world_size = world_size
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.db_path = self.data_dir / "jobs.sqlite3"
        self.default_prompt = default_prompt(args)
        self.default_image_path = default_image_path(args)
        self.default_image_bytes = self.default_image_path.read_bytes()
        self.default_image_suffix = self.default_image_path.suffix.lower()
        self.db_lock = threading.RLock()
        self.pending = queue.Queue()
        self.active_job_id = None
        self.ready = True
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._initialize_database()
        self._recover_jobs()

    def _initialize_database(self):
        with self.db_lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    queue_order INTEGER NOT NULL UNIQUE,
                    request_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    task TEXT,
                    prompt TEXT NOT NULL,
                    size TEXT,
                    frame_num INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    sample_shift REAL,
                    timesteps_index TEXT,
                    image_path TEXT NOT NULL,
                    action_path TEXT,
                    input_sources TEXT,
                    output_path TEXT NOT NULL,
                    job_dir TEXT NOT NULL,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            columns = {
                row["name"] for row in self.db.execute("PRAGMA table_info(jobs)")
            }
            if "queue_order" not in columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN queue_order INTEGER")
                rows = self.db.execute(
                    "SELECT id FROM jobs ORDER BY created_at, id"
                ).fetchall()
                for queue_order, row in enumerate(rows, start=1):
                    self.db.execute(
                        "UPDATE jobs SET queue_order = ? WHERE id = ?",
                        (queue_order, row["id"]),
                    )
            if "action_path" not in columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN action_path TEXT")
            migrations = {
                "task": "TEXT",
                "size": "TEXT",
                "sample_shift": "REAL",
                "timesteps_index": "TEXT",
                "input_sources": "TEXT",
            }
            for column, column_type in migrations.items():
                if column not in columns:
                    self.db.execute(
                        f"ALTER TABLE jobs ADD COLUMN {column} {column_type}"
                    )
            self.db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_queue_order_idx "
                "ON jobs(queue_order)"
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created_idx "
                "ON jobs(status, created_at)"
            )
            self.db.commit()

    def _missing_input(self, row):
        if not Path(row["image_path"]).is_file():
            return "input image is missing"
        if row["action_path"]:
            action_path = Path(row["action_path"])
            for filename in ("poses.npy", "intrinsics.npy"):
                if not (action_path / filename).is_file():
                    return f"uploaded trajectory file is missing: {filename}"
        return None

    def _recover_jobs(self):
        now = utc_timestamp()
        with self.db_lock:
            running_jobs = self.db.execute(
                "SELECT * FROM jobs WHERE status = 'running'"
            ).fetchall()
            for row in running_jobs:
                output_path = Path(row["output_path"])
                missing_input = self._missing_input(row)
                if output_path.is_file() and output_path.stat().st_size > 0:
                    self.db.execute(
                        "UPDATE jobs SET status = 'succeeded', finished_at = ?, error = NULL "
                        "WHERE id = ?",
                        (now, row["id"]),
                    )
                elif missing_input is None:
                    self.db.execute(
                        "UPDATE jobs SET status = 'queued', started_at = NULL, "
                        "error = 'recovered after server restart' WHERE id = ?",
                        (row["id"],),
                    )
                else:
                    self.db.execute(
                        "UPDATE jobs SET status = 'failed', finished_at = ?, "
                        "error = ? WHERE id = ?",
                        (
                            now,
                            f"{missing_input} during restart recovery",
                            row["id"],
                        ),
                    )

            queued_jobs = self.db.execute(
                "SELECT * FROM jobs WHERE status = 'queued' "
                "ORDER BY queue_order"
            ).fetchall()
            for row in queued_jobs:
                missing_input = self._missing_input(row)
                if missing_input is None:
                    self.pending.put(row["id"])
                else:
                    self.db.execute(
                        "UPDATE jobs SET status = 'failed', finished_at = ?, "
                        "error = ? WHERE id = ?",
                        (now, f"queued {missing_input}", row["id"]),
                    )
            self.db.commit()

    def _queue_position(self, row):
        if row["status"] != "queued":
            return None
        result = self.db.execute(
            """
            SELECT COUNT(*) AS position
            FROM jobs
            WHERE status = 'queued'
              AND queue_order <= ?
            """,
            (row["queue_order"],),
        ).fetchone()
        return result["position"]

    def _public_record(self, row):
        input_sources = (
            json.loads(row["input_sources"])
            if row["input_sources"]
            else {
                "prompt": "client",
                "image": "client",
                "poses.npy": "client" if row["action_path"] else "default",
                "intrinsics.npy": "client" if row["action_path"] else "default",
            }
        )
        array_sources = [
            source
            for filename, _, _, _ in ARRAY_INPUTS
            if (source := input_sources.get(filename)) not in (None, "unavailable")
        ]
        if not any(source == "client" for source in array_sources):
            trajectory_source = "default"
        elif all(source == "client" for source in array_sources):
            trajectory_source = "uploaded"
        else:
            trajectory_source = "mixed"
        return {
            "id": row["id"],
            "request_id": row["request_id"],
            "status": row["status"],
            "queue_position": self._queue_position(row),
            "task": row["task"] or self.args.task,
            "prompt": row["prompt"],
            "size": row["size"] or self.args.size,
            "frame_num": row["frame_num"],
            "seed": row["seed"],
            "sample_shift": row["sample_shift"]
            if row["sample_shift"] is not None
            else self.args.sample_shift,
            "timesteps_index": json.loads(row["timesteps_index"])
            if row["timesteps_index"]
            else list(self.args.timesteps_index),
            "input_sources": input_sources,
            "trajectory_source": trajectory_source,
            "attempts": row["attempts"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "result_path": row["output_path"] if row["status"] == "succeeded" else None,
            "video_url": f"/v1/jobs/{row['id']}/video"
            if row["status"] == "succeeded"
            else None,
        }

    def create_job(
        self,
        task,
        prompt,
        image_bytes,
        image_suffix,
        input_files,
        input_sources,
        size,
        frame_num,
        seed,
        sample_shift,
        timesteps_index,
        request_id,
    ):
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError(
                "request_id must contain only letters, digits, '.', '_', ':', or '-'"
            )

        with self.db_lock:
            existing = self.db.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                return self._public_record(existing), False

            queued_count = self.db.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = 'queued'"
            ).fetchone()["count"]
            if queued_count >= self.args.max_queue_size:
                raise QueueCapacityError(
                    f"queue is full ({self.args.max_queue_size} waiting jobs)"
                )

            job_id = uuid.uuid4().hex[:16]
            queue_order = self.db.execute(
                "SELECT COALESCE(MAX(queue_order), 0) + 1 AS next_order FROM jobs"
            ).fetchone()["next_order"]
            created_at = utc_timestamp()
            date_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            job_dir = self.jobs_dir / date_path / job_id
            image_path = job_dir / f"input{image_suffix}"
            output_path = self.output_dir / date_path / f"{job_id}.mp4"
            request_path = job_dir / "request.json"
            action_path = job_dir / "inputs"
            job_dir.mkdir(parents=True, exist_ok=False)
            try:
                image_path.write_bytes(image_bytes)
                action_path.mkdir()
                for filename, content in input_files.items():
                    (action_path / filename).write_bytes(content)
                request_path.write_text(
                    json.dumps(
                        {
                            "id": job_id,
                            "request_id": request_id,
                            "task": task,
                            "prompt": prompt,
                            "size": size,
                            "frame_num": frame_num,
                            "seed": seed,
                            "sample_shift": sample_shift,
                            "timesteps_index": timesteps_index,
                            "input_sources": input_sources,
                            "created_at": created_at,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self.db.execute(
                    """
                    INSERT INTO jobs (
                        id, queue_order, request_id, status, task, prompt, size,
                        frame_num, seed, sample_shift, timesteps_index, image_path,
                        action_path, input_sources, output_path, job_dir, created_at
                    ) VALUES (
                        :id, :queue_order, :request_id, 'queued', :task, :prompt,
                        :size, :frame_num, :seed, :sample_shift, :timesteps_index,
                        :image_path, :action_path, :input_sources, :output_path,
                        :job_dir, :created_at
                    )
                    """,
                    {
                        "id": job_id,
                        "queue_order": queue_order,
                        "request_id": request_id,
                        "task": task,
                        "prompt": prompt,
                        "size": size,
                        "frame_num": frame_num,
                        "seed": seed,
                        "sample_shift": sample_shift,
                        "timesteps_index": json.dumps(timesteps_index),
                        "image_path": str(image_path),
                        "action_path": str(action_path),
                        "input_sources": json.dumps(input_sources),
                        "output_path": str(output_path),
                        "job_dir": str(job_dir),
                        "created_at": created_at,
                    },
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise

            self.pending.put(job_id)
            row = self.db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return self._public_record(row), True

    def get_job(self, job_id):
        with self.db_lock:
            row = self.db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return self._public_record(row) if row is not None else None

    def list_jobs(self, limit):
        with self.db_lock:
            rows = self.db.execute(
                "SELECT * FROM jobs ORDER BY queue_order DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._public_record(row) for row in rows]

    def get_command(self, job_id):
        with self.db_lock:
            row = self.db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "type": "generate",
                "job_id": row["id"],
                "task": row["task"] or self.args.task,
                "prompt": row["prompt"],
                "image_path": row["image_path"],
                "action_path": row["action_path"] or self.args.action_path,
                "size": row["size"] or self.args.size,
                "frame_num": row["frame_num"],
                "seed": row["seed"],
                "sample_shift": row["sample_shift"]
                if row["sample_shift"] is not None
                else self.args.sample_shift,
                "timesteps_index": json.loads(row["timesteps_index"])
                if row["timesteps_index"]
                else list(self.args.timesteps_index),
                "output_path": row["output_path"],
            }

    def mark_running(self, job_id):
        with self.db_lock:
            cursor = self.db.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, finished_at = NULL,
                    error = NULL, attempts = attempts + 1
                WHERE id = ? AND status = 'queued'
                """,
                (utc_timestamp(), job_id),
            )
            self.db.commit()
            if cursor.rowcount != 1:
                return False
            self.active_job_id = job_id
            return True

    def mark_succeeded(self, job_id):
        with self.db_lock:
            self.db.execute(
                "UPDATE jobs SET status = 'succeeded', finished_at = ?, error = NULL "
                "WHERE id = ?",
                (utc_timestamp(), job_id),
            )
            self.db.commit()
            self.active_job_id = None

    def mark_failed(self, job_id, error):
        with self.db_lock:
            self.db.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error = ? "
                "WHERE id = ?",
                (utc_timestamp(), error, job_id),
            )
            self.db.commit()
            self.active_job_id = None

    def counts(self):
        with self.db_lock:
            rows = self.db.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
            counts = {row["status"]: row["count"] for row in rows}
            return {
                "queued": counts.get("queued", 0),
                "running": counts.get("running", 0),
                "succeeded": counts.get("succeeded", 0),
                "failed": counts.get("failed", 0),
            }

    def capabilities(self):
        default_inputs = {
            "prompt": True,
            "image": self.default_image_path.is_file(),
        }
        for filename, _, _, required in ARRAY_INPUTS:
            default_inputs[filename] = {
                "available": (Path(self.args.action_path) / filename).is_file(),
                "required": required,
            }
        return {
            "task": self.args.task,
            "request_parameters": {
                "sizes": list(SUPPORTED_SIZES),
                "default_size": self.args.size,
                "default_frame_num": self.args.frame_num,
                "max_frame_num": self.args.max_frame_num,
                "default_seed": 42,
                "default_sample_shift": self.args.sample_shift,
                "default_timesteps_index": list(self.args.timesteps_index),
            },
            "default_inputs": default_inputs,
            "server_parameters": {
                "checkpoint": Path(self.args.ckpt_dir).name,
                "infer_mode": "causal_fast",
                "dit_fsdp": self.args.dit_fsdp,
                "t5_fsdp": self.args.t5_fsdp,
                "ulysses_size": self.args.ulysses_size,
                "chunk_size": self.args.chunk_size,
                "local_attn_size": self.args.local_attn_size,
                "sink_size": self.args.sink_size,
                "offload_model": False,
                "max_attention_size": self.args.max_attention_size,
            },
            "max_upload_mb": self.args.max_upload_mb,
        }

    def video_path(self, job_id):
        with self.db_lock:
            row = self.db.execute(
                "SELECT status, output_path FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != "succeeded":
                return None
            return Path(row["output_path"])

    def cleanup_expired(self):
        if self.args.retention_hours == 0:
            return 0
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=self.args.retention_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.db_lock:
            rows = self.db.execute(
                """
                SELECT id, job_dir, output_path FROM jobs
                WHERE status IN ('succeeded', 'failed')
                  AND finished_at IS NOT NULL
                  AND finished_at < ?
                """,
                (cutoff,),
            ).fetchall()
            for row in rows:
                shutil.rmtree(row["job_dir"], ignore_errors=True)
                Path(row["output_path"]).unlink(missing_ok=True)
                self.db.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))
            self.db.commit()
            return len(rows)

    def close(self):
        with self.db_lock:
            self.db.close()


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    service = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ready" if self.service.ready else "stopping",
                    "world_size": self.service.world_size,
                    "active_job_id": self.service.active_job_id,
                    "jobs": self.service.counts(),
                    "max_queue_size": self.service.args.max_queue_size,
                    "data_dir": str(self.service.data_dir),
                    "output_dir": str(self.service.output_dir),
                },
            )
            return

        if path == "/v1/capabilities":
            self.send_json(HTTPStatus.OK, self.service.capabilities())
            return

        if path == "/v1/jobs":
            try:
                limit = int(parse_qs(parsed.query).get("limit", ["50"])[0])
                if not 1 <= limit <= 100:
                    raise ValueError
            except ValueError:
                self.send_error_json(
                    HTTPStatus.BAD_REQUEST, "limit must be between 1 and 100"
                )
                return
            self.send_json(HTTPStatus.OK, {"jobs": self.service.list_jobs(limit)})
            return

        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
            job = self.service.get_job(parts[2])
            if job is None:
                self.send_error_json(HTTPStatus.NOT_FOUND, "job not found")
            else:
                self.send_json(HTTPStatus.OK, job)
            return

        if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "video":
            self.send_video(parts[2])
            return

        self.send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_POST(self):
        if urlparse(self.path).path != "/v1/jobs":
            self.send_error_json(HTTPStatus.NOT_FOUND, "endpoint not found")
            return
        if not self.service.ready:
            self.send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, "service is stopping")
            return
        try:
            payload = self.read_json()
            task = payload.get("task") or self.service.args.task
            if not isinstance(task, str):
                raise ValueError("task must be a string")
            if task != self.service.args.task:
                raise ValueError(
                    f"task {task!r} is not loaded; server task is "
                    f"{self.service.args.task!r}"
                )

            prompt = payload.get("prompt")
            prompt_source = "client"
            if prompt is None:
                prompt = self.service.default_prompt
                prompt_source = "default"
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            if len(prompt) > 20_000:
                raise ValueError("prompt is too long")

            size = payload.get("size") or self.service.args.size
            if size not in SUPPORTED_SIZES:
                raise ValueError(
                    f"size must be one of: {', '.join(SUPPORTED_SIZES)}"
                )

            frame_num = payload.get("frame_num", self.service.args.frame_num)
            if isinstance(frame_num, bool) or not isinstance(frame_num, int):
                raise ValueError("frame_num must be an integer")
            if not 1 <= frame_num <= self.service.args.max_frame_num:
                raise ValueError(
                    f"frame_num must be between 1 and {self.service.args.max_frame_num}"
                )

            seed = payload.get("seed", 42)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")

            sample_shift = payload.get(
                "sample_shift", self.service.args.sample_shift
            )
            if (
                isinstance(sample_shift, bool)
                or not isinstance(sample_shift, (int, float))
                or not math.isfinite(sample_shift)
                or sample_shift <= 0
            ):
                raise ValueError("sample_shift must be a positive finite number")
            sample_shift = float(sample_shift)

            timesteps_index = payload.get(
                "timesteps_index", list(self.service.args.timesteps_index)
            )
            self.validate_timesteps(timesteps_index)

            request_id = payload.get("request_id") or uuid.uuid4().hex
            if not isinstance(request_id, str):
                raise ValueError("request_id must be a string")
            image_bytes, image_suffix, image_source = self.decode_image(payload)
            input_files, array_sources = self.decode_trajectory(payload)
            input_sources = {
                "prompt": prompt_source,
                "image": image_source,
                **array_sources,
            }
            job, created = self.service.create_job(
                task,
                prompt.strip(),
                image_bytes,
                image_suffix,
                input_files,
                input_sources,
                size,
                frame_num,
                seed,
                sample_shift,
                timesteps_index,
                request_id,
            )
            self.send_json(HTTPStatus.ACCEPTED if created else HTTPStatus.OK, job)
        except QueueCapacityError as exc:
            self.send_error_json(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
        except (ValueError, binascii.Error, UnidentifiedImageError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            logging.exception("Failed to submit job")
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "failed to submit job"
            )

    def read_json(self):
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Content-Length is required")
        try:
            content_length = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc

        max_file_bytes = self.service.args.max_upload_mb * 1024 * 1024
        # One image and five arrays, all base64 encoded.
        max_body_bytes = max_file_bytes * 8 + 1024 * 1024
        if content_length < 1 or content_length > max_body_bytes:
            raise ValueError("request body is too large")
        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def decode_image(self, payload):
        encoded = payload.get("image_base64")
        if encoded is None:
            if "image_name" in payload:
                raise ValueError("image_base64 is required when image_name is provided")
            return (
                self.service.default_image_bytes,
                self.service.default_image_suffix,
                "default",
            )
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_base64 must be a non-empty string")
        image_bytes = base64.b64decode(encoded, validate=True)
        max_bytes = self.service.args.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise ValueError("uploaded image is too large")

        image_name = payload.get("image_name", "image.jpg")
        suffix = Path(image_name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError("image must be jpg, jpeg, png, or webp")

        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image.verify()
        if width < 1 or height < 1 or width * height > 50_000_000:
            raise ValueError("image dimensions are not supported")
        return image_bytes, suffix, "client"

    def decode_trajectory(self, payload):
        trajectory = payload.get("trajectory")
        if trajectory is None:
            trajectory = {}
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be a JSON object")

        max_bytes = self.service.args.max_upload_mb * 1024 * 1024
        files = {}
        arrays = {}
        sources = {}
        default_input_dir = Path(self.service.args.action_path)
        for filename, field, expected_tail, required in ARRAY_INPUTS:
            encoded = trajectory.get(field)
            if encoded is None:
                default_path = default_input_dir / filename
                if not default_path.is_file():
                    if required:
                        raise ValueError(f"default input file is missing: {filename}")
                    sources[filename] = "unavailable"
                    continue
                content = default_path.read_bytes()
                sources[filename] = "default"
            else:
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError(f"trajectory.{field} must be non-empty")
                content = base64.b64decode(encoded, validate=True)
                sources[filename] = "client"
            if len(content) > max_bytes:
                raise ValueError(f"input array is too large: {filename}")
            try:
                array = np.load(io.BytesIO(content), allow_pickle=False)
            except (EOFError, OSError, ValueError) as exc:
                raise ValueError(f"invalid input array: {filename}") from exc
            if not isinstance(array, np.ndarray):
                if hasattr(array, "close"):
                    array.close()
                raise ValueError(f"input file must contain one array: {filename}")
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"input array must contain finite numbers: {filename}")
            if array.ndim != len(expected_tail) + 1 or array.shape[1:] != expected_tail:
                shape = ", ".join(("frames", *(str(item) for item in expected_tail)))
                raise ValueError(f"{filename} must have shape ({shape})")
            files[filename] = content
            arrays[filename] = array

        poses = arrays["poses.npy"]
        intrinsics = arrays["intrinsics.npy"]
        if poses.shape[0] != intrinsics.shape[0]:
            raise ValueError("trajectory arrays must have the same frame count")
        minimum_frames = 4 * (self.service.args.chunk_size - 1) + 1
        if poses.shape[0] < minimum_frames:
            raise ValueError(
                f"trajectory must contain at least {minimum_frames} frames"
            )
        return files, sources

    def validate_timesteps(self, timesteps):
        if not isinstance(timesteps, list) or not 1 <= len(timesteps) <= 16:
            raise ValueError("timesteps_index must be a list of 1 to 16 integers")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in timesteps
        ):
            raise ValueError("timesteps_index must contain only integers")
        if any(item < 0 or item >= 1000 for item in timesteps):
            raise ValueError("timesteps_index values must be between 0 and 999")
        if any(left >= right for left, right in zip(timesteps, timesteps[1:])):
            raise ValueError("timesteps_index must be strictly increasing")

    def send_video(self, job_id):
        path = self.service.video_path(job_id)
        if path is None:
            job = self.service.get_job(job_id)
            if job is None:
                self.send_error_json(HTTPStatus.NOT_FOUND, "job not found")
            else:
                self.send_error_json(HTTPStatus.CONFLICT, "video is not ready")
            return
        try:
            video_file = path.open("rb")
        except OSError:
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, "video file is missing"
            )
            return

        with video_file:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header(
                "Content-Length", str(os.fstat(video_file.fileno()).st_size)
            )
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            self.end_headers()
            try:
                while chunk := video_file.read(1024 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json(status, {"error": message})

    def log_message(self, format_string, *args):
        logging.info("HTTP %s - %s", self.address_string(), format_string % args)


class InferenceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def validate_server_paths(args):
    checkpoint_dir = Path(args.ckpt_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    action_path = Path(args.action_path)
    for filename in ("poses.npy", "intrinsics.npy"):
        path = action_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"camera file not found: {path}")
    image_path = default_image_path(args)
    if not image_path.is_file():
        raise FileNotFoundError(f"default image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported default image type: {image_path.suffix}")
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    if width < 1 or height < 1 or width * height > 50_000_000:
        raise ValueError("default image dimensions are not supported")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "jobs").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def create_pipeline(args, rank, local_rank, world_size):
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.distributed.util import init_distributed_group

    config = WAN_CONFIGS[args.task]
    if world_size != args.ulysses_size:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must equal --ulysses-size ({args.ulysses_size})"
        )
    if config.num_heads % args.ulysses_size:
        raise ValueError("model heads must be divisible by --ulysses-size")
    init_distributed_group()

    logging.info("Loading persistent inference pipeline on rank %s", rank)
    pipeline = wan.WanI2VCausal(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=True,
        t5_cpu=False,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode="causal_fast",
    )
    return pipeline, config


def run_generation(pipeline, config, args, command, rank):
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.utils import save_video

    image = Image.open(command["image_path"]).convert("RGB")
    video = None
    try:
        if command["task"] != args.task:
            raise ValueError(
                f"job task {command['task']!r} does not match loaded task {args.task!r}"
            )
        video = pipeline.generate(
            command["prompt"],
            image,
            action_path=command["action_path"],
            chunk_size=args.chunk_size,
            max_area=MAX_AREA_CONFIGS[command["size"]],
            frame_num=command["frame_num"],
            timesteps_index=command["timesteps_index"],
            shift=command["sample_shift"],
            seed=command["seed"],
            offload_model=False,
            max_attention_size=args.max_attention_size,
        )
        if rank == 0:
            output_path = Path(command["output_path"])
            partial_path = output_path.with_name("result.partial.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.unlink(missing_ok=True)
            try:
                save_video(
                    tensor=video[None],
                    save_file=str(partial_path),
                    fps=config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if not partial_path.is_file() or partial_path.stat().st_size == 0:
                    raise RuntimeError("video encoder did not create an output file")
                os.replace(partial_path, output_path)
            finally:
                partial_path.unlink(missing_ok=True)
    finally:
        image.close()
        del video
        if hasattr(pipeline, "self_kv_cache"):
            pipeline.self_kv_cache = None
        gc.collect()
        torch.cuda.empty_cache()


def execute_generation(pipeline, config, args, command, rank):
    local_error = None
    try:
        run_generation(pipeline, config, args, command, rank)
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        logging.error("Generation failed on rank %s:\n%s", rank, traceback.format_exc())

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return [error for error in errors if error is not None]


def worker_loop(pipeline, config, args, rank):
    while True:
        payload = [None]
        dist.broadcast_object_list(payload, src=0)
        command = payload[0]
        if command["type"] == "shutdown":
            return
        if command["type"] == "heartbeat":
            continue
        execute_generation(pipeline, config, args, command, rank)


def rank_zero_loop(pipeline, config, args, world_size):
    state = ServiceState(args, world_size)
    ApiHandler.service = state
    httpd = None
    http_thread = None

    try:
        httpd = InferenceHTTPServer((args.host, args.port), ApiHandler)
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        logging.info("Server ready at http://%s:%s", args.host, args.port)
        logging.info("Persistent job data: %s", state.data_dir)
        logging.info("Generated video output: %s", state.output_dir)

        while True:
            try:
                job_id = state.pending.get(timeout=30)
            except queue.Empty:
                dist.broadcast_object_list([{"type": "heartbeat"}], src=0)
                cleaned = state.cleanup_expired()
                if cleaned:
                    logging.info("Removed %s expired jobs", cleaned)
                continue
            if not state.mark_running(job_id):
                continue
            command = state.get_command(job_id)
            dist.broadcast_object_list([command], src=0)
            errors = execute_generation(pipeline, config, args, command, rank=0)
            if not errors:
                state.mark_succeeded(job_id)
                logging.info("Job %s succeeded", job_id)
            else:
                error_message = "; ".join(errors)
                logging.error("Job %s failed: %s", job_id, error_message)
                state.mark_failed(job_id, error_message)
            cleaned = state.cleanup_expired()
            if cleaned:
                logging.info("Removed %s expired jobs", cleaned)
    except KeyboardInterrupt:
        logging.info("Stopping server")
    finally:
        state.ready = False
        try:
            dist.broadcast_object_list([{"type": "shutdown"}], src=0)
        except Exception:
            logging.exception("Could not notify workers during shutdown")
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if http_thread is not None:
            http_thread.join(timeout=5)
        state.close()


def init_logging(rank):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format=f"[%(asctime)s] [rank {rank}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_logging(rank)

    if world_size < 2:
        raise RuntimeError("Start server.py with torchrun --nproc_per_node=8")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )
    try:
        if rank == 0:
            validate_server_paths(args)
        dist.barrier()
        pipeline, config = create_pipeline(args, rank, local_rank, world_size)
        dist.barrier()
        if rank == 0:
            rank_zero_loop(pipeline, config, args, world_size)
        else:
            worker_loop(pipeline, config, args, rank)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
