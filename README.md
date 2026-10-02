# Gas Pipeline Maintenance Rover ? Baseline V1

Software-only pipeline inspection baseline. It supports the six-class Pipeline Defect Dataset (Deformation, Obstacle, Rupture, Disconnect, Misalignment, Deposition), YOLO training/evaluation, image/video inference, temporal confirmation, duplicate suppression, simulated distance, SQLite events, confirmed-image storage, replay, dashboard, and HTML reports. Existing project description is preserved in this expanded documentation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python training/dataset_audit.py
```

The audit is read-only and writes `reports/dataset_audit.json` and `.csv`. The 22,120 JPEG images are stored under `data/raw/train` via Git LFS, but the supplied copy contains no YOLO label files, so no class distribution is inferred and training is blocked until labels are supplied. Dataset facts are recorded in `data/dataset_manifest.json`. Use `scripts/download_dataset.py` for Kaggle (credentials are required).

## Training and evaluation

Prepare/verify `data/dataset.yaml`, then run `python training/train.py [--model yolo26s.pt --epochs 100 --imgsz 640]`. Ultralytics is imported only when training/evaluation/inference is requested, with clear errors when unavailable. Evaluation writes only actual metrics to `reports/evaluation_metrics.json`; weights and run artifacts remain ignored.

## Replay pipeline

```bash
python simulation/video_replay.py --source inspection.mp4 --weights models/best.pt
python reports/generate_report.py --session SESSION_ID --database inspection/SESSION_ID/events.db
streamlit run dashboard/app.py
```

A detection must persist for the configured number of frames (default three). One best-confidence frame is saved only after confirmation. Events explicitly use `distance_source: simulation`; no GPIO or Raspberry Pi dependency exists. `Obstacle` is detected, displayed, and logged but never drives motors. Interfaces in `simulation/hardware.py` are ready for later camera, encoder, and ultrasonic implementations.

## Layout

- `training/`: audit, train, evaluate
- `inference/`: detector adapter, IoU tracker, confirmation/event manager
- `simulation/`: encoder and video replay
- `storage/`: SQLite and confirmed image store
- `dashboard/`, `reports/`, `scripts/`, `tests/`

Run tests with `python -m unittest discover -s tests` and compile checks with `python -m compileall .`. Raw data, models, caches, inspection output, secrets, and virtual environments are excluded by `.gitignore`.
