"""Runtime environment helpers for third-party libraries."""

from __future__ import annotations

import os
from pathlib import Path


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def ensure_local_runtime_home(base_dir: Path | None = None) -> Path:
    """Point HOME/USERPROFILE/ARVIZ_DATA to a writable project-local directory."""

    project_root = base_dir or Path(__file__).resolve().parents[3]
    runtime_home = project_root / ".runtime_home"
    arviz_dir = runtime_home / "arviz_data"

    current_home = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or "")
    if current_home and _is_writable_directory(current_home / "arviz_data"):
        os.environ.setdefault("ARVIZ_DATA", str(current_home / "arviz_data"))
        return current_home

    runtime_home.mkdir(parents=True, exist_ok=True)
    arviz_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    os.environ["USERPROFILE"] = str(runtime_home)
    os.environ["ARVIZ_DATA"] = str(arviz_dir)
    return runtime_home
