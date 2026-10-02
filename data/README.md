# Local dataset integration

The supplied Kaggle extraction is included in this repository through Git LFS.
It is approximately 1.2 GB and contains 22,120 JPEG images under `data/raw/train`.
The default `config/config.yaml` and `data/dataset.yaml` point to this layout.

The current extraction contains no YOLO annotation files. The audit therefore
reports zero annotations and training refuses to start until the matching
six-class annotation files are added. Do not infer or fabricate labels from
the images.

Run:

```powershell
python training/dataset_audit.py
```

The tracked dataset facts are recorded in `data/dataset_manifest.json`.
