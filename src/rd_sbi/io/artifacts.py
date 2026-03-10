"""Artifact naming and metadata utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp_compact() -> str:
    """Return UTC timestamp in YYYYMMDDTHHMMSSZ format."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_artifact_name(task_id: str, run_id: str, extension: str, timestamp_utc: str | None = None) -> str:
    """Build artifact name from policy."""
    ts = timestamp_utc or utc_timestamp_compact()
    ext = extension.lstrip(".")
    return f"{task_id}__{run_id}__{ts}.{ext}"


def write_metadata_sidecar(artifact_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write JSON metadata sidecar next to an artifact."""
    path = Path(artifact_path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    payload = {
        "artifact": path.name,
        "created_utc": utc_timestamp_compact(),
        **metadata,
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar
