from pathlib import Path

import torch

from llm_trainer.train.checkpoint import load_checkpoint, save_checkpoint


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)



def test_checkpoint_roundtrip(tmp_path: Path):
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ckpt = tmp_path / "step_1.pt"

    save_checkpoint(
        ckpt,
        model,
        optimizer,
        scheduler,
        global_step=1,
        epoch=0,
        seen_tokens=128,
        config_snapshot={"foo": "bar"},
        tokenizer_ref="tok.model",
        tokenizer_hash="deadbeef",
    )

    restored = load_checkpoint(ckpt, model, optimizer, scheduler)
    assert restored["global_step"] == 1
    assert restored["seen_tokens"] == 128
    assert restored["tokenizer_ref"] == "tok.model"
