#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/.cache /app/models

if [ "$#" -eq 0 ]; then
  exec python -m llm_trainer --help
fi

exec "$@"
