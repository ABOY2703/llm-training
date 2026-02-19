from __future__ import annotations

import hashlib
import re
from pathlib import Path

import sentencepiece as spm


def _corpus_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _chunk_for_sentencepiece(text: str, max_chars: int = 4000) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    parts: list[str] = []
    # Prefer splitting by sentence boundaries first, then hard-chunk overflow.
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            parts.append(sentence)
            continue
        for i in range(0, len(sentence), max_chars):
            parts.append(sentence[i : i + max_chars])
    return parts


def train_sentencepiece(
    texts: list[str],
    out_dir: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    model_prefix: str = "tokenizer",
) -> tuple[Path, Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as f:
        for t in texts:
            for chunk in _chunk_for_sentencepiece(t, max_chars=4000):
                f.write(chunk + "\n")

    prefix_path = out_dir / model_prefix
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    model_path = out_dir / f"{model_prefix}.model"
    vocab_path = out_dir / f"{model_prefix}.vocab"
    return model_path, vocab_path, _file_hash(model_path)


def load_processor(model_path: Path) -> spm.SentencePieceProcessor:
    proc = spm.SentencePieceProcessor()
    proc.load(str(model_path))
    return proc


def encode_texts(processor: spm.SentencePieceProcessor, texts: list[str]) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(processor.encode(text, out_type=int))
        token_ids.append(processor.eos_id())
    return token_ids
