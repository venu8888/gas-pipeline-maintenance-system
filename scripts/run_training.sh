#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

python training/dataset_audit.py --root data/raw
python training/train.py --model yolov8s.pt --epochs 50 --imgsz 640 --data data/dataset.yaml
