#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python prepare_assets.py
python task1.py --input_path examples/task1 --output_path results/runs/task1.png
python task2.py --input_path examples/task2 --output_path results/runs/task2.png --json results/runs/task2_overlap.json
