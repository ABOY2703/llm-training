FROM python:3.11-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY llm_trainer /app/llm_trainer
COPY configs /app/configs
COPY docker/entrypoint.sh /entrypoint.sh

RUN pip install --upgrade pip && \
    pip install -e .

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "llm_trainer", "--help"]


FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY pyproject.toml README.md /app/
COPY llm_trainer /app/llm_trainer
COPY configs /app/configs
COPY docker/entrypoint.sh /entrypoint.sh

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch && \
    python -m pip install -e . --no-deps && \
    python -m pip install datasets sentencepiece transformers PyYAML tqdm

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "llm_trainer", "--help"]
