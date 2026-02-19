from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    cache_dir: str = ".cache"
    run_name: str = "default_run"
    txt: list[str] = field(default_factory=list)
    hf: list[str] = field(default_factory=list)
    hf_config: str | None = None
    hf_split: str = "train"
    hf_text_column: str | None = None
    seed: int = 42
    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 1
    seq_len: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int | None = None
    checkpoint_every: int = 200
    vocab_size: int = 8000
    spm_model_type: str = "bpe"
    spm_character_coverage: float = 1.0
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    metrics_log_file: str = "metrics.jsonl"
    retokenize: bool = False

    def __post_init__(self) -> None:
        int_fields = [
            "seed",
            "epochs",
            "batch_size",
            "grad_accum_steps",
            "seq_len",
            "warmup_steps",
            "checkpoint_every",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ]
        float_fields = [
            "lr",
            "weight_decay",
            "spm_character_coverage",
            "rms_norm_eps",
            "rope_theta",
        ]
        for name in int_fields:
            setattr(self, name, int(getattr(self, name)))
        for name in float_fields:
            setattr(self, name, float(getattr(self, name)))

        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def merge_config(cli: dict[str, Any], yaml_values: dict[str, Any], require_dataset: bool = True) -> TrainConfig:
    defaults = TrainConfig().__dict__.copy()
    merged = {**defaults, **yaml_values}
    for key, value in cli.items():
        if value is not None:
            merged[key] = value
    cfg = TrainConfig(**merged)
    if require_dataset and not cfg.txt and not cfg.hf:
        raise ValueError("At least one dataset source required: --txt and/or --hf")
    return cfg
