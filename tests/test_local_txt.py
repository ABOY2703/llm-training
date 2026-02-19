from pathlib import Path

from llm_trainer.data.local_txt import load_local_texts


def test_load_local_texts_supports_absolute_file(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("hello", encoding="utf-8")

    texts = load_local_texts([str(f)])
    assert texts == ["hello"]


def test_load_local_texts_supports_absolute_glob(tmp_path: Path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("a", encoding="utf-8")
    (d / "b.txt").write_text("b", encoding="utf-8")

    pattern = str(d / "*.txt")
    texts = load_local_texts([pattern])
    assert sorted(texts) == ["a", "b"]
