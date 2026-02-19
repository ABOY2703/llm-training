from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    out_path: Path,
    model,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    seen_tokens: int,
    config_snapshot: dict,
    tokenizer_ref: str,
    tokenizer_hash: str,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "global_step": global_step,
        "epoch": epoch,
        "seen_tokens": seen_tokens,
        "config_snapshot": config_snapshot,
        "tokenizer_ref": tokenizer_ref,
        "tokenizer_hash": tokenizer_hash,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def save_run_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
