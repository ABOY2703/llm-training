from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


CANDIDATE_TEXT_COLUMNS = ["text", "content", "document", "body"]


def _select_text_column(columns: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Requested --hf-text-column '{explicit}' not found in {columns}")
        return explicit
    for candidate in CANDIDATE_TEXT_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(f"No text-like column found. Columns: {columns}")


def load_hf_texts(
    names: list[str],
    split: str,
    cache_dir: Path,
    hf_config: str | None = None,
    text_column: str | None = None,
) -> list[str]:
    all_texts: list[str] = []
    for name in names:
        ds = load_dataset(name, hf_config, split=split, cache_dir=str(cache_dir))
        columns = list(ds.features.keys())
        col = _select_text_column(columns, text_column)
        for item in ds:
            value = item.get(col)
            if isinstance(value, str) and value.strip():
                all_texts.append(value)
    return all_texts
