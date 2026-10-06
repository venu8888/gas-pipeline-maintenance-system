"""Train YOLO on the local dataset in the repo's data/ directory."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from training.dataset_audit import audit

NAMES = ['Deformation', 'Obstacle', 'Rupture', 'Disconnect', 'Misalignment', 'Deposition']
ROOT = Path(__file__).resolve().parents[1]


def require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise RuntimeError('Training requires ultralytics. Install requirements.txt first.') from exc


def resolve_training_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Training requires PyTorch. Install the dependencies from requirements.txt.') from exc

    if requested.lower() == 'auto':
        return '0' if torch.cuda.is_available() else 'cpu'
    if requested.lower() == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'GPU training was requested (device={requested}), but this environment has CPU-only PyTorch. '
            'Install a CUDA-enabled PyTorch build in .venv, then retry. '
            'For CPU-only training, pass --device cpu.'
        )
    return requested


def resolve_dataset_root(config_path: str | Path) -> Path:
    config = load_config(config_path)
    yaml_path = Path(config['dataset']['yaml'])
    if not yaml_path.is_absolute():
        yaml_path = ROOT / yaml_path
    if not yaml_path.exists():
        raise FileNotFoundError(f'Dataset YAML not found: {yaml_path}.')

    dataset_root = Path(config['dataset']['root'])
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    candidates = [dataset_root, ROOT / 'data']
    for candidate in candidates:
        if candidate.exists() and audit(candidate)['total_annotations'] > 0:
            return candidate

    raise FileNotFoundError(f'Dataset root not found: {dataset_root}.')


def sanitize_invalid_boxes(dataset_root: Path) -> dict[str, int]:
    fixed_files = 0
    fixed_annotations = 0
    for label_file in sorted(dataset_root.rglob('*.txt')):
        if 'train.cache' in label_file.parts:
            continue
        lines = label_file.read_text(encoding='utf-8', errors='replace').splitlines()
        cleaned: list[str] = []
        invalid_count = 0
        for raw in lines:
            if not raw.strip():
                continue
            parts = raw.strip().split()
            if len(parts) != 5:
                cleaned.append(raw)
                continue
            try:
                cid = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:]]
            except ValueError:
                cleaned.append(raw)
                continue
            if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
                invalid_count += 1
                fixed_annotations += 1
                continue
            cleaned.append(f'{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
        if invalid_count:
            label_file.write_text('\n'.join(cleaned) + ('\n' if cleaned else ''), encoding='utf-8')
            fixed_files += 1
    return {'fixed_files': fixed_files, 'fixed_annotations': fixed_annotations}


def prepare_yolo_label_layout(data_yaml: Path, dataset_root: Path) -> int:
    try:
        from ultralytics.data.utils import check_det_dataset, img2label_paths
    except ImportError as exc:
        raise RuntimeError('Training requires ultralytics. Install requirements.txt first.') from exc

    dataset = check_det_dataset(str(data_yaml))
    split_paths: list[str] = []
    for split_name in ('train', 'val', 'test'):
        split = dataset.get(split_name)
        if split:
            split_paths.extend(split if isinstance(split, list) else [split])
    images: set[Path] = set()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for split_path in split_paths:
        path = Path(split_path)
        if path.is_file() and path.suffix.lower() == '.txt':
            entries = (
                line.strip()
                for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            )
            for entry in entries:
                image = Path(entry)
                if not image.is_absolute():
                    image = path.parent / image
                if image.suffix.lower() in image_extensions:
                    images.add(image.resolve())
        elif path.is_dir():
            images.update(
                image.resolve()
                for image in path.rglob('*')
                if image.is_file() and image.suffix.lower() in image_extensions
            )
    images = sorted(images)
    source_labels: dict[str, Path] = {}
    for label in sorted((dataset_root / 'labels').rglob('*.txt')):
        if label.stem in source_labels:
            raise RuntimeError(f'Duplicate source label filename found: {label.stem}.txt')
        source_labels[label.stem] = label

    expected_labels = img2label_paths([str(image) for image in images])
    missing = [image for image, label in zip(images, expected_labels) if image.stem not in source_labels]
    if missing:
        sample = ', '.join(str(path) for path in missing[:5])
        raise RuntimeError(f'{len(missing)} training images have no matching source label. Examples: {sample}')

    copied = 0
    cache_paths: set[Path] = set()
    for image, target_text in zip(images, expected_labels):
        source = source_labels[image.stem]
        target = Path(target_text)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
            copied += 1
        cache_paths.add(target.parent.with_suffix('.cache'))

    for cache_path in cache_paths:
        if cache_path.exists():
            cache_path.unlink()
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--imgsz', type=int)
    ap.add_argument('--data', default=None)
    ap.add_argument('--batch', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--cache', choices=('ram', 'disk', 'none'), default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--resume-from', type=Path, help='Resume from a specific Ultralytics last.pt checkpoint.')
    ap.add_argument('--config', default='config/config.yaml')
    a = ap.parse_args()

    c = load_config(a.config)
    device = resolve_training_device(str(a.device if a.device is not None else c['training'].get('device', 'auto')))
    data_path = Path(a.data or c['dataset']['yaml'])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset YAML not found: {data_path}.')

    dataset_root = resolve_dataset_root(a.config)
    audit_json = Path('reports/dataset_audit.json')
    if audit_json.exists():
        try:
            audit_result = json.loads(audit_json.read_text(encoding='utf-8'))
            print('[1/3] Loaded existing dataset audit from reports/dataset_audit.json.')
        except Exception:
            print('[1/3] Auditing dataset...')
            audit_result = audit(dataset_root)
    else:
        print('[1/3] Auditing dataset (first run)...')
        audit_result = audit(dataset_root)

    if audit_result['total_annotations'] == 0:
        raise RuntimeError('No annotations were found in the local dataset. Check data/images/images/train and data/labels/labels/train.')
    if audit_result['invalid_class_ids'] or audit_result['malformed_labels']:
        raise RuntimeError('Dataset contains invalid class IDs or malformed annotations; fix the labels before training.')

    sanitized = sanitize_invalid_boxes(dataset_root)
    if sanitized['fixed_annotations']:
        audit_result = audit(dataset_root)
        print(f'Sanitized {sanitized["fixed_annotations"]} invalid boxes across {sanitized["fixed_files"]} label files.')

    if not audit_result['mapping_matches_expected'] or audit_result['total_annotations'] == 0:
        raise RuntimeError('Dataset class mapping/annotation check failed after sanitization. Review reports/dataset_audit.json.')

    print('[2/3] Preparing YOLO dataset layout...')
    copied_labels = prepare_yolo_label_layout(data_path, dataset_root)
    if copied_labels:
        print(f'Prepared {copied_labels} YOLO label files beside the image dataset and cleared stale label caches.')

    print('[3/3] Initializing YOLO training...')

    if a.resume_from and a.model:
        raise ValueError('Use either --resume-from or --model, not both.')
    if a.resume_from and a.resume:
        raise ValueError('Use --resume-from alone; it enables resume automatically.')
    checkpoint = a.resume_from or (Path(a.model) if a.resume and a.model else None)
    if a.resume and checkpoint is None:
        raise ValueError('Resume requires a checkpoint: use --resume-from PATH or --model PATH --resume.')
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f'Resume checkpoint not found: {checkpoint}')
    weights = str(checkpoint or a.model or c['model']['weights'])
    epochs = a.epochs or c['training']['epochs']
    imgsz = a.imgsz or c['training']['image_size']
    batch_size = a.batch if a.batch is not None else c['training']['batch_size']
    workers = a.workers if a.workers is not None else c['training'].get('workers', 0)
    if sys.platform == 'win32' and workers > 0:
        print(f"[Notice] Automatically overriding workers={workers} -> workers=0 on Windows to prevent PyTorch DataLoader multiprocessing deadlocks.")
        workers = 0
    cache = a.cache if a.cache is not None else c['training'].get('cache', False)
    if cache == 'none':
        cache = False

    YOLO = require_ultralytics()
    model = YOLO(weights)
    kwargs = {
        'data': str(data_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': -1 if str(batch_size).lower() == 'auto' else int(batch_size),
        'device': device,
        'workers': workers,
        'cache': cache,
        'amp': c['training'].get('amp', True),
        'patience': c['training']['patience'],
        'pretrained': c['training']['pretrained'],
        'seed': c['training']['seed'],
        'save_period': c['training'].get('save_period', 5),
        'project': 'training_outputs',
        'name': f'{Path(weights).stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}',
        'exist_ok': False,
        'resume': bool(checkpoint),
    }

    results = model.train(**kwargs)
    save_dir = Path(getattr(results, 'save_dir', kwargs['project']))
    weights_dir = save_dir / 'weights'
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    for source, target in ((weights_dir / 'best.pt', models_dir / 'best.pt'), (weights_dir / 'last.pt', models_dir / 'last.pt')):
        if source.exists():
            shutil.copy2(source, target)

    best_epoch = getattr(model.trainer, 'best', None)
    if hasattr(best_epoch, 'epoch'):
        best_epoch = best_epoch.epoch
    elif isinstance(best_epoch, dict):
        best_epoch = best_epoch.get('epoch')

    out = Path('reports')
    out.mkdir(exist_ok=True)
    record = {
        'experiment_id': kwargs['name'],
        'model': weights,
        'dataset': str(data_path),
        'epochs': epochs,
        'image_size': imgsz,
        'batch_size': kwargs['batch'],
        'best_epoch': best_epoch,
        'training_date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'classes': NAMES,
    }
    try:
        import yaml
        (out / 'training_config.yaml').write_text(yaml.safe_dump(record, sort_keys=False), encoding='utf-8')
    except ImportError:
        (out / 'training_config.yaml').write_text(json.dumps(record, indent=2), encoding='utf-8')

    print(results)


if __name__ == '__main__':
    main()
