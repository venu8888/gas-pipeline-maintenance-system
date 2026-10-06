# Gas Pipeline Maintenance Rover

This repository is configured for the local dataset you added under `data/`:

- Images: `data/images/images/train`
- Labels: `data/labels/labels/train`
- Dataset config: `data/dataset.yaml`

The project trains a YOLO model for six defect classes:

- Deformation
- Obstacle
- Rupture
- Disconnect
- Misalignment
- Deposition

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python training/dataset_audit.py --root data
python training/train.py --config config/config.yaml
```

The dataset audit checks the actual label/image pairing and class mapping and writes `reports/dataset_audit.json` and `.csv`.

## Training and evaluation

The project uses the local dataset directly and keeps the model checkpoints in `models/` and experiment output in `training_outputs/`.

```powershell
python training\split_dataset.py
python training\train.py --epochs 100 --imgsz 640 --batch auto --device 0 --workers 4
python training\evaluate.py --weights models\best.pt --data data\dataset.yaml --split test
```

The split command writes deterministic 70/20/10 train/validation/test image manifests without copying the image files. It uses seed 42 by default. Ultralytics updates `last.pt` at each completed epoch and retains `best.pt`; extra numbered epoch copies are disabled to reduce checkpoint I/O. To resume from a checkpoint, pass `--resume-from training_outputs\<run>\weights\last.pt`. Recovery is at the latest saved epoch boundary, not an arbitrary batch within an epoch. GPU training requires a CUDA-enabled PyTorch build; the script stops with an actionable error rather than silently using the CPU when GPU device `0` is requested. Automatic batch sizing, mixed precision, and four data-loader workers are enabled for GPU training. If sufficient system RAM is available, `--cache ram` can speed up image loading; it is opt-in because the memory requirement depends on image dimensions. The script also records the best epoch in the training metadata. Use `--device cpu` only to explicitly run on the CPU.

## Rover-specific requirements

This repo still includes simulation support for a rover that can:

- capture a camera frame,
- compute distance travelled from an encoder/sensor simulation,
- store only confirmed defect events and distance metadata in SQLite,
- alert via the event pipeline when a defect is confirmed.

The simulated hardware does not require a physical Raspberry Pi or ultrasonic sensor to run.

## Layout

- `training/`: audit, train, evaluate
- `inference/`: detector adapter, IoU tracker, confirmation/event manager
- `simulation/`: encoder and video replay
- `storage/`: SQLite and confirmed image store
- `dashboard/`, `reports/`, `scripts/`, `tests/`

Run tests with `python -m unittest discover -s tests` and compile checks with `python -m compileall .`.
