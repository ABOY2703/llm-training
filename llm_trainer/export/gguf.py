from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")


def export_gguf(hf_dir: Path, out_path: Path, llama_cpp_dir: Path | None = None, quantize: str | None = None) -> Path:
    if llama_cpp_dir is None:
        llama_cpp_dir = Path("llama.cpp")
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(
            f"Could not find converter script at {converter}. Clone llama.cpp or pass --llama-cpp-dir"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(converter),
            str(hf_dir),
            "--outfile",
            str(out_path),
            "--outtype",
            "f16",
        ]
    )

    if quantize:
        quant_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not quant_bin.exists():
            raise FileNotFoundError(
                f"Quantize binary not found at {quant_bin}. Build llama.cpp or omit --quantize"
            )
        quant_out = out_path.with_name(f"{out_path.stem}.{quantize}.gguf")
        _run([str(quant_bin), str(out_path), str(quant_out), quantize])
        return quant_out

    if not out_path.exists():
        raise FileNotFoundError(f"Expected GGUF output was not created: {out_path}")
    return out_path
