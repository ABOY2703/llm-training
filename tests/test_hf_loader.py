from pathlib import Path

from llm_trainer.data.hf_loader import load_hf_texts


class DummyDS:
    def __init__(self):
        self.features = {"text": str, "id": int}
        self.rows = [{"text": "a"}, {"text": "b"}]

    def __iter__(self):
        return iter(self.rows)



def test_load_hf_texts(monkeypatch, tmp_path: Path):
    def fake_load_dataset(name, cfg, split, cache_dir):
        assert name == "demo/ds"
        assert split == "train"
        assert cache_dir == str(tmp_path)
        return DummyDS()

    monkeypatch.setattr("llm_trainer.data.hf_loader.load_dataset", fake_load_dataset)
    texts = load_hf_texts(["demo/ds"], split="train", cache_dir=tmp_path)
    assert texts == ["a", "b"]
