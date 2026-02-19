from llm_trainer.tokenizer.spm import _chunk_for_sentencepiece


def test_chunk_for_sentencepiece_splits_very_long_line():
    text = "a" * 10000
    chunks = _chunk_for_sentencepiece(text, max_chars=4000)
    assert len(chunks) == 3
    assert max(len(c) for c in chunks) <= 4000


def test_chunk_for_sentencepiece_keeps_short_text():
    text = "hello world"
    chunks = _chunk_for_sentencepiece(text, max_chars=4000)
    assert chunks == ["hello world"]
