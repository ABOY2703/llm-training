from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from llm_trainer.config import TrainConfig
from llm_trainer.train.checkpoint import save_checkpoint, save_run_state


def compute_rates(elapsed_s: float, token_count: int) -> tuple[float, float]:
    elapsed = max(elapsed_s, 1e-9)
    return 1.0 / elapsed, token_count / elapsed


def make_batches(token_ids: list[int], seq_len: int) -> TensorDataset:
    if len(token_ids) < seq_len + 1:
        raise ValueError("Not enough tokens for one training sequence. Add more data or lower --seq-len")
    n_full = len(token_ids) // (seq_len + 1)
    usable = n_full * (seq_len + 1)
    arr = torch.tensor(token_ids[:usable], dtype=torch.long)
    arr = arr.view(n_full, seq_len + 1)
    x = arr[:, :-1]
    y = arr[:, 1:]
    return TensorDataset(x, y)


def run_training(
    cfg: TrainConfig,
    model,
    token_ids: list[int],
    run_paths: dict[str, Path],
    tokenizer_ref: str,
    tokenizer_hash: str,
    resume_state: dict | None = None,
) -> dict:
    device = next(model.parameters()).device
    dataset = make_batches(token_ids, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    total_updates = math.ceil((len(loader) * cfg.epochs) / cfg.grad_accum_steps)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=max(1, total_updates),
    )

    global_step = 0
    start_epoch = 0
    seen_tokens = 0
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if resume_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        seen_tokens = int(resume_state.get("seen_tokens", 0))

    model.train()
    metrics_path = run_paths["run"] / cfg.metrics_log_file
    with metrics_path.open("a", encoding="utf-8") as metrics_f:
        for epoch in range(start_epoch, cfg.epochs):
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", unit="step")
            last_t = time.perf_counter()
            for step_idx, (input_ids, labels) in enumerate(bar, start=1):
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                try:
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out.loss / cfg.grad_accum_steps
                    loss.backward()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device.type in {"mps", "cuda"}:
                        backend = device.type.upper()
                        raise RuntimeError(
                            f"{backend} OOM during training. Lower --batch-size/--seq-len or increase --grad-accum-steps."
                        ) from e
                    raise

                if step_idx % cfg.grad_accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    now = time.perf_counter()
                    dt = now - last_t
                    last_t = now
                    tokens_this_step = input_ids.numel() * cfg.grad_accum_steps
                    seen_tokens += tokens_this_step
                    iter_per_sec, tok_per_sec = compute_rates(dt, tokens_this_step)
                    lr = scheduler.get_last_lr()[0]
                    loss_scalar = float(loss.item() * cfg.grad_accum_steps)

                    bar.set_postfix(
                        loss=f"{loss_scalar:.4f}",
                        it_s=f"{iter_per_sec:.2f}",
                        tok_s=f"{tok_per_sec:.0f}",
                        lr=f"{lr:.2e}",
                    )

                    log_row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": loss_scalar,
                        "iter_per_sec": iter_per_sec,
                        "tokens_per_sec": tok_per_sec,
                        "seen_tokens": seen_tokens,
                        "lr": lr,
                    }
                    metrics_f.write(json.dumps(log_row) + "\n")
                    metrics_f.flush()

                    if global_step % cfg.checkpoint_every == 0:
                        ckpt_path = run_paths["checkpoints"] / f"step_{global_step}.pt"
                        save_checkpoint(
                            ckpt_path,
                            model,
                            optimizer,
                            scheduler,
                            global_step=global_step,
                            epoch=epoch,
                            seen_tokens=seen_tokens,
                            config_snapshot=cfg.__dict__,
                            tokenizer_ref=tokenizer_ref,
                            tokenizer_hash=tokenizer_hash,
                        )

                    if cfg.max_steps and global_step >= cfg.max_steps:
                        break

            epoch_ckpt = run_paths["checkpoints"] / f"step_{global_step}.pt"
            save_checkpoint(
                epoch_ckpt,
                model,
                optimizer,
                scheduler,
                global_step=global_step,
                epoch=epoch + 1,
                seen_tokens=seen_tokens,
                config_snapshot=cfg.__dict__,
                tokenizer_ref=tokenizer_ref,
                tokenizer_hash=tokenizer_hash,
            )
            if cfg.max_steps and global_step >= cfg.max_steps:
                break

    final_path = run_paths["run"] / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    save_run_state(
        run_paths["meta"] / "run_state.json",
        {
            "run_name": cfg.run_name,
            "global_step": global_step,
            "epoch": cfg.epochs,
            "seen_tokens": seen_tokens,
            "final_model": str(final_path),
            "tokenizer_ref": tokenizer_ref,
            "tokenizer_hash": tokenizer_hash,
            "metrics": str(metrics_path),
        },
    )
    return {
        "global_step": global_step,
        "seen_tokens": seen_tokens,
        "final_model": str(final_path),
    }
