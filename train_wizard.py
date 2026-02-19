#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def _must_int(value: str, field: str, min_value: int = 1) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < min_value:
        raise ValueError(f"{field} must be >= {min_value}")
    return parsed


def _build_command(
    txt_path: str | None,
    epochs: int,
    run_name: str,
    hf_dataset: str | None,
    hf_split: str,
    hf_config: str | None,
    hf_text_column: str | None,
    batch_size: int,
    seq_len: int,
    lr: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "llm_trainer",
        "train",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--lr",
        str(lr),
        "--run-name",
        run_name,
    ]
    if txt_path:
        cmd.extend(["--txt", txt_path])
    if hf_dataset:
        cmd.extend(["--hf", hf_dataset, "--hf-split", hf_split])
        if hf_config:
            cmd.extend(["--hf-config", hf_config])
        if hf_text_column:
            cmd.extend(["--hf-text-column", hf_text_column])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive non-docker trainer launcher")
    parser.add_argument("--yes", action="store_true", help="Use defaults where possible without extra prompts")
    args = parser.parse_args()

    print("LLM Training Wizard (local Python, no Docker)")

    source_mode = _ask("Data source: local / hf / both", "local").strip().lower()
    if source_mode not in {"local", "hf", "both"}:
        raise ValueError("Data source must be one of: local, hf, both")

    txt_path: str | None = None
    if source_mode in {"local", "both"}:
        default_txt = str(Path.home() / "Downloads" / "wiki_text_data.txt")
        txt_path = _ask("Path to local .txt file", default_txt)
        if not Path(txt_path).expanduser().exists():
            raise FileNotFoundError(f"Local text file not found: {txt_path}")

    epochs_value = _ask("How many epochs", "1")
    epochs = _must_int(epochs_value, "epochs")

    hf_dataset = None
    hf_split = "train"
    hf_config = None
    hf_text_column = None
    if source_mode in {"hf", "both"}:
        hf_dataset = _ask("Hugging Face dataset name", "dtian09/wiki_text_data")
        hf_split = _ask("Hugging Face split", "train")
        hf_config_value = _ask("Hugging Face dataset config (leave empty for none)", "")
        hf_text_column_value = _ask("Hugging Face text column (leave empty for auto-detect)", "")
        hf_config = hf_config_value or None
        hf_text_column = hf_text_column_value or None

    run_default = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = _ask("Run name", run_default)

    if args.yes:
        batch_size = 4
        seq_len = 256
        lr = 3e-4
    else:
        batch_size = _must_int(_ask("Batch size", "4"), "batch_size")
        seq_len = _must_int(_ask("Sequence length", "256"), "seq_len")
        lr = float(_ask("Learning rate", "3e-4"))

    cmd = _build_command(
        txt_path=txt_path,
        epochs=epochs,
        run_name=run_name,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_config=hf_config,
        hf_text_column=hf_text_column,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
    )

    print("\nRunning command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print()

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
