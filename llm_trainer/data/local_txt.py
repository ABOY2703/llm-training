from __future__ import annotations

import glob
from pathlib import Path


def load_local_texts(patterns: list[str]) -> list[str]:
    texts: list[str] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        for path in matches:
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts
