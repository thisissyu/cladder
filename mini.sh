#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/mini

uv run fig generate config/demo-mini.yml \
    --path runs/mini/questions.json \
    --model-meta runs/mini/models.json \
    --spec-limit 1 \
    --graph-cap 1 \
    --question-limit 1 \
    --overwrite true \
    --summarize false

uv run python mini.py
