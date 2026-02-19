from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_trainer.config import TrainConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: TrainConfig, vocab_size: int) -> LlamaForCausalLM:
    model_cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    return LlamaForCausalLM(model_cfg)
