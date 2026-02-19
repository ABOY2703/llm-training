from __future__ import annotations

import hashlib
import json
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_fingerprint(payload: dict) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def cache_paths(cache_dir: Path, run_name: str) -> dict[str, Path]:
    base = ensure_dir(cache_dir)
    run_dir = ensure_dir(base / "runs" / run_name)
    checkpoints = ensure_dir(run_dir / "checkpoints")
    meta = ensure_dir(run_dir / "meta")
    dataset_cache = ensure_dir(base / "datasets")
    tokenizer_cache = ensure_dir(base / "tokenizers" / run_name)
    export_dir = ensure_dir(run_dir / "export")
    return {
        "base": base,
        "run": run_dir,
        "checkpoints": checkpoints,
        "meta": meta,
        "datasets": dataset_cache,
        "tokenizer": tokenizer_cache,
        "export": export_dir,
    }
