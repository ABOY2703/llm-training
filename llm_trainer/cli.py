from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch

from llm_trainer.config import TrainConfig, load_yaml, merge_config
from llm_trainer.data.cache import cache_paths, source_fingerprint
from llm_trainer.data.hf_loader import load_hf_texts
from llm_trainer.data.local_txt import load_local_texts
from llm_trainer.export.gguf import export_gguf
from llm_trainer.model.transformer import build_model, get_device
from llm_trainer.tokenizer.spm import encode_texts, load_processor, train_sentencepiece
from llm_trainer.train.checkpoint import find_latest_checkpoint, load_checkpoint
from llm_trainer.train.loop import run_training


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _collect_texts(cfg: TrainConfig, dataset_cache: Path) -> list[str]:
    texts: list[str] = []
    if cfg.txt:
        texts.extend(load_local_texts(cfg.txt))
    if cfg.hf:
        texts.extend(
            load_hf_texts(
                cfg.hf,
                split=cfg.hf_split,
                cache_dir=dataset_cache,
                hf_config=cfg.hf_config,
                text_column=cfg.hf_text_column,
            )
        )
    if not texts:
        raise ValueError("No text samples loaded from datasets")
    return texts


def _dataset_manifest(cfg: TrainConfig) -> dict:
    return {
        "txt": sorted(cfg.txt),
        "hf": sorted(cfg.hf),
        "hf_config": cfg.hf_config,
        "hf_split": cfg.hf_split,
        "hf_text_column": cfg.hf_text_column,
    }


def _prepare_tokenizer(cfg: TrainConfig, texts: list[str], run_paths: dict[str, Path]) -> tuple[Path, str, list[int]]:
    model_path, _, corpus_hash = train_sentencepiece(
        texts,
        run_paths["tokenizer"],
        vocab_size=cfg.vocab_size,
        model_type=cfg.spm_model_type,
        character_coverage=cfg.spm_character_coverage,
    )
    proc = load_processor(model_path)
    token_ids = encode_texts(proc, texts)
    return model_path, corpus_hash, token_ids


def _reuse_tokenizer(tokenizer_path: Path, texts: list[str]) -> tuple[str, list[int]]:
    proc = load_processor(tokenizer_path)
    token_ids = encode_texts(proc, texts)
    tok_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()[:16]
    return tok_hash, token_ids


def _save_hf_artifacts(model, run_paths: dict[str, Path], tokenizer_model: Path) -> Path:
    hf_dir = run_paths["export"] / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir)
    shutil.copy2(tokenizer_model, hf_dir / "tokenizer.model")
    return hf_dir


def _build_config(args: argparse.Namespace, require_dataset: bool = True) -> TrainConfig:
    yaml_values = load_yaml(Path(args.config) if args.config else None)
    cli_values = {
        "cache_dir": args.cache_dir,
        "run_name": args.run_name,
        "txt": args.txt,
        "hf": args.hf,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_text_column": args.hf_text_column,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "checkpoint_every": args.checkpoint_every,
        "vocab_size": args.vocab_size,
        "spm_model_type": args.spm_model_type,
        "spm_character_coverage": args.spm_character_coverage,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "metrics_log_file": args.metrics_log_file,
        "retokenize": args.retokenize,
    }
    return merge_config(cli_values, yaml_values, require_dataset=require_dataset)


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    _set_seed(cfg.seed)

    run_paths = cache_paths(Path(cfg.cache_dir), cfg.run_name)
    manifest = _dataset_manifest(cfg)
    manifest["fingerprint"] = source_fingerprint(manifest)
    (run_paths["meta"] / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    texts = _collect_texts(cfg, run_paths["datasets"])
    tokenizer_path, tokenizer_hash, token_ids = _prepare_tokenizer(cfg, texts, run_paths)

    model = build_model(cfg, vocab_size=cfg.vocab_size)
    device = get_device()
    model.to(device)

    summary = run_training(
        cfg,
        model,
        token_ids,
        run_paths,
        tokenizer_ref=str(tokenizer_path),
        tokenizer_hash=tokenizer_hash,
        resume_state=None,
    )

    hf_dir = _save_hf_artifacts(model, run_paths, tokenizer_path)
    print(f"Training completed. Global step: {summary['global_step']}")
    print(f"Final model state: {summary['final_model']}")
    print(f"HF export dir: {hf_dir}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    cfg = _build_config(args, require_dataset=False)
    _set_seed(cfg.seed)

    run_paths = cache_paths(Path(cfg.cache_dir), cfg.run_name)
    if args.latest:
        checkpoint_path = find_latest_checkpoint(run_paths["checkpoints"])
    elif args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        raise ValueError("resume requires --checkpoint or --latest")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_snapshot = raw_ckpt.get("config_snapshot", {})
    snapshot_cfg = TrainConfig(**{**TrainConfig().__dict__, **cfg_snapshot})

    # Preserve architecture from checkpoint. CLI controls training loop and data sources.
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
    ]:
        setattr(cfg, key, getattr(snapshot_cfg, key))

    tokenizer_ref = Path(raw_ckpt["tokenizer_ref"]) if "tokenizer_ref" in raw_ckpt else None
    if tokenizer_ref is None or not tokenizer_ref.exists():
        raise FileNotFoundError("Checkpoint tokenizer_ref missing or invalid")

    if not cfg.txt and not cfg.hf:
        manifest_path = run_paths["meta"] / "dataset_manifest.json"
        if not manifest_path.exists():
            raise ValueError("No dataset sources provided and no saved dataset manifest found for this run")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg.txt = manifest.get("txt", [])
        cfg.hf = manifest.get("hf", [])
        cfg.hf_config = cfg.hf_config or manifest.get("hf_config")
        cfg.hf_split = cfg.hf_split or manifest.get("hf_split")
        cfg.hf_text_column = cfg.hf_text_column or manifest.get("hf_text_column")

    texts = _collect_texts(cfg, run_paths["datasets"])
    if cfg.retokenize:
        tokenizer_path, tokenizer_hash, token_ids = _prepare_tokenizer(cfg, texts, run_paths)
    else:
        tokenizer_path = tokenizer_ref
        tokenizer_hash, token_ids = _reuse_tokenizer(tokenizer_path, texts)
        if str(raw_ckpt.get("tokenizer_hash")) != str(tokenizer_hash):
            raise ValueError(
                "Tokenizer mismatch on resume. Use --retokenize to explicitly override tokenizer continuity."
            )

    model = build_model(cfg, vocab_size=cfg.vocab_size)
    device = get_device()
    model.to(device)

    resume_state = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    )

    summary = run_training(
        cfg,
        model,
        token_ids,
        run_paths,
        tokenizer_ref=str(tokenizer_path),
        tokenizer_hash=str(tokenizer_hash),
        resume_state=resume_state,
    )

    hf_dir = _save_hf_artifacts(model, run_paths, tokenizer_path)
    print(f"Resume completed from: {checkpoint_path}")
    print(f"Global step: {summary['global_step']}")
    print(f"HF export dir: {hf_dir}")
    return 0


def cmd_export_gguf(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    hf_dir = run_dir / "export" / "hf"
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF export directory not found: {hf_dir}")

    out_path = Path(args.out)
    gguf_path = export_gguf(
        hf_dir=hf_dir,
        out_path=out_path,
        llama_cpp_dir=Path(args.llama_cpp_dir) if args.llama_cpp_dir else None,
        quantize=args.quantize,
    )
    size_mb = gguf_path.stat().st_size / (1024 * 1024)
    print(f"GGUF exported: {gguf_path} ({size_mb:.2f} MB)")
    return 0


def _add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--txt", nargs="*", default=None)
    p.add_argument("--hf", nargs="*", default=None)
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default=None)
    p.add_argument("--hf-text-column", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--spm-model-type", default=None)
    p.add_argument("--spm-character-coverage", type=float, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--num-hidden-layers", type=int, default=None)
    p.add_argument("--num-attention-heads", type=int, default=None)
    p.add_argument("--num-key-value-heads", type=int, default=None)
    p.add_argument("--rms-norm-eps", type=float, default=None)
    p.add_argument("--rope-theta", type=float, default=None)
    p.add_argument("--metrics-log-file", default=None)
    p.add_argument("--retokenize", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train from scratch")
    _add_common_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_resume = sub.add_parser("resume", help="Resume training from checkpoint")
    _add_common_train_args(p_resume)
    p_resume.add_argument("--checkpoint", default=None)
    p_resume.add_argument("--latest", action="store_true")
    p_resume.set_defaults(func=cmd_resume)

    p_export = sub.add_parser("export-gguf", help="Export GGUF from run artifacts")
    p_export.add_argument("--run", required=True)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--llama-cpp-dir", default=None)
    p_export.add_argument("--quantize", default=None)
    p_export.set_defaults(func=cmd_export_gguf)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
