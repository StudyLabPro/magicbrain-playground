"""Очередь задач плейграунда.

Прогон длится от секунд до пары минут, поэтому HTTP-запрос его не ждёт:
задача уходит в пул потоков, клиент опрашивает статус. numpy освобождает GIL
на тяжёлых операциях, так что потоков достаточно — процессы дали бы только
лишний расход памяти при жёстком лимите контейнера.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import settings


class Progress:
    """Канал прогресса, который эксперимент дёргает по ходу работы."""

    def __init__(self, job: dict[str, Any], lock: threading.Lock):
        self._job = job
        self._lock = lock

    def __call__(self, fraction: float, message: str = "", **extra: Any) -> None:
        with self._lock:
            self._job["progress"] = max(0.0, min(1.0, float(fraction)))
            if message:
                self._job["message"] = message
            if extra:
                self._job.setdefault("detail", {}).update(extra)
            self._job["updated_at"] = time.time()

    def stage(self, name: str, index: int, total: int) -> None:
        self(index / max(1, total), f"этап {index}/{total}: {name}", stage=name)


class JobRunner:
    def __init__(self, max_workers: int | None = None):
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or settings.MAX_WORKERS,
            thread_name_prefix="mbplay",
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Progress], dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        job = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "message": "в очереди",
            "params": params,
            "created_at": time.time(),
            "updated_at": time.time(),
            "run_id": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._trim()
        progress = Progress(job, self._lock)

        def _run() -> None:
            with self._lock:
                job["status"] = "running"
                job["started_at"] = time.time()
                job["message"] = "запущено"
            try:
                run = fn(progress)
                with self._lock:
                    job["status"] = "done"
                    job["progress"] = 1.0
                    job["message"] = "готово"
                    job["run_id"] = run.get("run_id")
                    job["headline"] = run.get("headline", {})
            except Exception as exc:  # noqa: BLE001 — ошибку показываем в UI целиком
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = f"{type(exc).__name__}: {exc}"
                    job["traceback"] = traceback.format_exc(limit=8)
                    job["message"] = "ошибка"
            finally:
                with self._lock:
                    job["finished_at"] = time.time()
                    job["updated_at"] = time.time()

        self._pool.submit(_run)
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            return [
                {k: v for k, v in job.items() if k not in ("params", "traceback")}
                for job in jobs[:limit]
            ]

    def _trim(self, keep: int = 200) -> None:
        if len(self._jobs) <= keep:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j["status"] in ("done", "failed")),
            key=lambda j: j.get("finished_at", 0),
        )
        for job in finished[: len(self._jobs) - keep]:
            self._jobs.pop(job["job_id"], None)


runner = JobRunner()
