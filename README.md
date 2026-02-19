# llm_training

Terminal-based LLM training from scratch with:
- Local `.txt` and Hugging Face datasets
- Progress bar with `it/s`, `tokens/s`, `loss`, and `lr`
- Resume training with full state restoration
- Project-local cache in `.cache/`
- GGUF export via llama.cpp conversion tooling

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

## Quick start

```bash
python3 -m llm_trainer train \
  --txt data/*.txt \
  --hf dtian09/wiki_text_data \
  --hf-split train \
  --epochs 3 \
  --batch-size 8 \
  --seq-len 256 \
  --lr 3e-4 \
  --run-name wiki_exp1
```

Interactive local launcher (single file, no Docker):

```bash
python3 train_wizard.py
```

`train_wizard.py` asks whether you want `local`, `hf`, or `both`, then prompts only for the needed fields.

Resume from latest checkpoint:

```bash
python3 -m llm_trainer resume --run-name wiki_exp1 --latest --epochs 2
```

Export GGUF:

```bash
python3 -m llm_trainer export-gguf \
  --run .cache/runs/wiki_exp1 \
  --out models/wiki_exp1.gguf
```

## Docker

One `Dockerfile` provides two targets:
- `cpu`: portable CPU image (macOS/Linux hosts).
- `cuda`: NVIDIA CUDA image for GPU hosts (e.g. dual T4).

Build CPU image:

```bash
docker build -f Dockerfile --target cpu -t llm-trainer:cpu .
```

Run CPU training:

```bash
docker run --rm -it \
  -v "$PWD":/app \
  -v "$PWD/.cache":/app/.cache \
  llm-trainer:cpu \
  python -m llm_trainer train --txt data/*.txt --epochs 1 --run-name cpu_run
```

Build CUDA image:

```bash
docker build -f Dockerfile --target cuda -t llm-trainer:cuda .
```

Run CUDA training:

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/app \
  -v "$PWD/.cache":/app/.cache \
  llm-trainer:cuda \
  python -m llm_trainer train --txt data/*.txt --epochs 1 --run-name gpu_run
```

Auto-detect CPU vs CUDA image and run the right one:

```bash
./docker/run-auto.sh python -m llm_trainer --help
./docker/run-auto.sh python -m llm_trainer train --txt data/*.txt --epochs 1 --run-name auto_run
```

Using docker compose:

```bash
docker compose build llm-trainer-cpu
docker compose run --rm llm-trainer-cpu python -m llm_trainer --help
docker compose build llm-trainer-cuda
docker compose run --rm llm-trainer-cuda python -m llm_trainer --help
```

## Google Colab (T4 / dual T4)

In Colab you typically run directly (not Docker). Use:

```bash
!pip install -e '.[dev]'
!python -m llm_trainer train --txt /content/your.txt --epochs 1 --batch-size 8 --run-name colab_run
```

## Notes

- Runtime now supports CUDA, Apple Silicon (`mps`), and CPU fallback.
- Apple `mps` acceleration works in local native Python runs. Docker on macOS uses Linux containers and should be treated as CPU unless a Linux GPU runtime is available.
- GGUF export requires llama.cpp conversion tools locally available.
- Default config: `configs/default.yaml`.
