from pathlib import Path

from llm_trainer.export.gguf import export_gguf


def test_export_gguf_with_fake_converter(tmp_path: Path):
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir(parents=True)
    converter = llama_dir / "convert_hf_to_gguf.py"
    converter.write_text(
        "import pathlib,sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outfile')+1])\n"
        "out.write_bytes(b'gguf')\n",
        encoding="utf-8",
    )

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()

    out = tmp_path / "model.gguf"
    produced = export_gguf(hf_dir=hf_dir, out_path=out, llama_cpp_dir=llama_dir)
    assert produced.exists()
    assert produced.read_bytes() == b"gguf"
