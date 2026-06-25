from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock
from typing import Any
from uuid import uuid4


_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = Lock()


def create_job(action: str, *, steps: list[str], run_id: int | None = None) -> dict[str, Any]:
    now = datetime.utcnow().isoformat()
    job_id = f"job_{uuid4().hex[:16]}"
    job = {
        "job_id": job_id,
        "action": action,
        "status": "queued",
        "run_id": run_id,
        "steps": steps,
        "step_index": 0,
        "current_step": steps[0] if steps else "Queued",
        "completed_items": 0,
        "total_items": 0,
        "detail": "Queued",
        "error": None,
        "result": None,
        "started_at": now,
        "updated_at": now,
    }
    with _LOCK:
        _JOBS[job_id] = job
    return get_job(job_id) or job


def update_job(job_id: str, **updates: Any) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        if "step_index" in updates and job.get("steps"):
            index = min(max(int(updates["step_index"]), 0), len(job["steps"]) - 1)
            job["current_step"] = job["steps"][index]
        return deepcopy(job)


def increment_job(job_id: str, *, detail: str | None = None) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        job["completed_items"] = int(job.get("completed_items") or 0) + 1
        if detail is not None:
            job["detail"] = detail
        job["updated_at"] = datetime.utcnow().isoformat()
        return deepcopy(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return deepcopy(job) if job else None
