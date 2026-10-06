"""Create reproducible, class-stratified YOLO train/validation/test manifests without copying images."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
CLASS_NAMES = {
    0: 'Deformation',
    1: 'Obstacle',
    2: 'Rupture',
    3: 'Disconnect',
    4: 'Misalignment',
    5: 'Deposition',
}


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 0 or not ratios or any(ratio < 0 for ratio in ratios.values()):
        raise ValueError('Total and split ratios must be non-negative, with at least one split.')
    ratio_sum = sum(ratios.values())
    if not 0.999999 <= ratio_sum <= 1.000001:
        raise ValueError(f'Split ratios must sum to 1.0; got {ratio_sum}.')

    exact = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(ratios, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def parse_label_file(label_path: str) -> tuple[set[int], Counter]:
    classes = set()
    counts = Counter()
    try:
        with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cid = int(float(parts[0]))
                        classes.add(cid)
                        counts[cid] += 1
                    except ValueError:
                        pass
    except OSError:
        pass
    return classes, counts


def stratified_multilabel_split(
    images: list[Path],
    image_classes: dict[Path, set[int]],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, list[Path]]:
    rnd = random.Random(seed)

    class_to_images: dict[int, list[Path]] = defaultdict(list)
    for img in images:
        cls_set = image_classes.get(img, set())
        if cls_set:
            for cid in cls_set:
                class_to_images[cid].append(img)
        else:
            class_to_images[-1].append(img)

    sorted_classes = sorted(class_to_images.keys(), key=lambda c: len(class_to_images[c]))

    assigned: set[Path] = set()
    split_manifests: dict[str, list[Path]] = {name: [] for name in ratios}

    for cid in sorted_classes:
        unassigned = [img for img in class_to_images[cid] if img not in assigned]
        if not unassigned:
            continue
        rnd.shuffle(unassigned)

        counts = split_counts(len(unassigned), ratios)
        cursor = 0
        for split_name in ratios:
            chunk = unassigned[cursor : cursor + counts[split_name]]
            cursor += counts[split_name]
            split_manifests[split_name].extend(chunk)
            assigned.update(chunk)

    remaining = [img for img in images if img not in assigned]
    if remaining:
        rnd.shuffle(remaining)
        counts = split_counts(len(remaining), ratios)
        cursor = 0
        for split_name in ratios:
            chunk = remaining[cursor : cursor + counts[split_name]]
            cursor += counts[split_name]
            split_manifests[split_name].extend(chunk)
            assigned.update(chunk)

    return split_manifests


def create_splits(
    data_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    stratified: bool = True,
) -> dict[str, object]:
    ratios = ratios or DEFAULT_RATIOS
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    image_dir = data_root / 'images' / 'images' / 'train'
    label_dir = data_root / 'labels' / 'labels' / 'train'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if not label_dir.is_dir():
        raise FileNotFoundError(f'Label directory not found: {label_dir}')

    images: list[Path] = []
    with os.scandir(image_dir) as entries:
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                images.append(Path(entry.path))
    images.sort()

    labels_by_stem: dict[str, str] = {}
    with os.scandir(label_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.txt'):
                stem = os.path.splitext(entry.name)[0]
                if stem in labels_by_stem:
                    raise ValueError(f'Duplicate label filename: {entry.name}')
                labels_by_stem[stem] = entry.path

    missing = [image.name for image in images if image.stem not in labels_by_stem]
    extra = sorted(set(labels_by_stem) - {image.stem for image in images})
    if missing or extra:
        raise ValueError(
            f'Image/label pairing failed: {len(missing)} images lack labels and '
            f'{len(extra)} labels lack images.'
        )
    if not images:
        raise ValueError(f'No images found under {image_dir}')

    image_classes: dict[Path, set[int]] = {}
    image_ann_counts: dict[Path, Counter] = {}

    for img in images:
        label_path = labels_by_stem.get(img.stem)
        cls_set, ann_cnt = parse_label_file(label_path) if label_path else (set(), Counter())
        image_classes[img] = cls_set
        image_ann_counts[img] = ann_cnt

    if stratified:
        raw_manifests = stratified_multilabel_split(images, image_classes, ratios, seed)
    else:
        counts = split_counts(len(images), ratios)
        shuffled = images.copy()
        random.Random(seed).shuffle(shuffled)
        raw_manifests = {}
        cursor = 0
        for name in ratios:
            raw_manifests[name] = shuffled[cursor : cursor + counts[name]]
            cursor += counts[name]

    manifests: dict[str, list[Path]] = {}
    split_class_stats: dict[str, dict[str, dict[str, int]]] = {}

    for name, img_list in raw_manifests.items():
        sorted_list = sorted(img_list)
        manifests[name] = sorted_list

        img_counter = Counter()
        ann_counter = Counter()
        for img in sorted_list:
            for cid in image_classes[img]:
                img_counter[cid] += 1
            for cid, cnt in image_ann_counts[img].items():
                ann_counter[cid] += cnt

        split_class_stats[name] = {
            'images_per_class': {
                CLASS_NAMES.get(cid, str(cid)): img_counter[cid] for cid in sorted(img_counter)
            },
            'annotations_per_class': {
                CLASS_NAMES.get(cid, str(cid)): ann_counter[cid] for cid in sorted(ann_counter)
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Write absolute paths so YOLO can resolve both images and labels correctly.
    # YOLO derives label paths by replacing the 'images' segment with 'labels',
    # which only works reliably with absolute paths on Windows.
    for name, entries in manifests.items():
        (output_dir / f'{name}.txt').write_text(
            ''.join(f'{image.resolve().as_posix()}\n' for image in entries),
            encoding='utf-8',
        )

    summary: dict[str, object] = {
        'seed': seed,
        'stratified': stratified,
        'ratios': ratios,
        'counts': {name: len(entries) for name, entries in manifests.items()},
        'total_images': len(images),
        'label_count': len(labels_by_stem),
        'split_files': {name: f'{name}.txt' for name in manifests},
        'per_split_class_distribution': split_class_stats,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/splits'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--random-only',
        action='store_true',
        help='Disable class-wise stratification and use unstratified random split.',
    )
    args = parser.parse_args()

    summary = create_splits(
        args.data_root,
        args.output_dir,
        seed=args.seed,
        stratified=not args.random_only,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
