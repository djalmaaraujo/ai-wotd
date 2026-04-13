"""Read/write state files under data/index/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def load_cursors(index_dir: Path) -> dict:
    return _read(index_dir / "state.json", {})


def save_cursors(index_dir: Path, cursors: dict) -> None:
    _write(index_dir / "state.json", cursors)


def load_processed_days(index_dir: Path) -> dict:
    return _read(index_dir / "processed_days.json", {})


def save_processed_days(index_dir: Path, data: dict) -> None:
    _write(index_dir / "processed_days.json", data)


def load_url_cache(index_dir: Path) -> dict:
    return _read(index_dir / "url_cache.json", {})


def save_url_cache(index_dir: Path, data: dict) -> None:
    _write(index_dir / "url_cache.json", data)


def load_term_first_seen(index_dir: Path) -> dict:
    return _read(index_dir / "term_first_seen.json", {})


def save_term_first_seen(index_dir: Path, data: dict) -> None:
    _write(index_dir / "term_first_seen.json", data)
