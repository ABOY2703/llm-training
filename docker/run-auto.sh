#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

has_nvidia_runtime() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia
}

ensure_image() {
  local image="$1"
  local target="$2"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Building missing image: $image (target=$target)"
    docker build -f Dockerfile --target "$target" -t "$image" .
  fi
}

if has_nvidia_runtime; then
  IMAGE="llm-trainer:cuda"
  TARGET="cuda"
  EXTRA_ARGS=(--gpus all)
  echo "Detected NVIDIA Docker runtime. Using CUDA image."
else
  IMAGE="llm-trainer:cpu"
  TARGET="cpu"
  EXTRA_ARGS=()
  echo "No NVIDIA Docker runtime detected. Using CPU image."
fi

ensure_image "$IMAGE" "$TARGET"

mkdir -p .cache models

docker run --rm -it \
  "${EXTRA_ARGS[@]}" \
  -v "$PWD":/app \
  -v "$PWD/.cache":/app/.cache \
  "$IMAGE" \
  "$@"
