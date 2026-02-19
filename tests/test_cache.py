from pathlib import Path

from llm_trainer.data.cache import cache_paths, source_fingerprint


def test_source_fingerprint_stable():
    payload = {"hf": ["a"], "txt": ["x.txt"]}
    fp1 = source_fingerprint(payload)
    fp2 = source_fingerprint(payload)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_cache_paths_created(tmp_path: Path):
    paths = cache_paths(tmp_path / ".cache", "run1")
    assert paths["run"].exists()
    assert paths["checkpoints"].exists()
    assert paths["datasets"].exists()
