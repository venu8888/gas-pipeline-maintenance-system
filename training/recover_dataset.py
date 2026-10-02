#!/usr/bin/env python3
"""Search for existing annotation sources and prepare a recoverable report.

This script does not fabricate labels. It only identifies valid annotation sources,
converts them when possible, and stops with a clear blocker when the repo lacks
usable YOLO labels for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
EXPECTED_CLASSES = {
    0: "Deformation",
    1: "Obstacle",
    2: "Rupture",
    3: "Disconnect",
    4: "Misalignment",
    5: "Deposition",
}


def find_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and ".git" not in p.parts and ".venv" not in p.parts
    )


def search_for_annotations(root: Path) -> dict[str, Any]:
    files = find_files(root, ANNOTATION_EXTS)
    by_ext = defaultdict(list)
    for path in files:
        by_ext[path.suffix.lower()].append(str(path.relative_to(root)))
    return {
        "total_annotation_files": len(files),
        "by_extension": {k: sorted(v) for k, v in sorted(by_ext.items())},
    }


def count_images(root: Path) -> list[Path]:
    return sorted(p for p in find_files(root, IMAGE_EXTS) if p.is_file())


def yaml_name_map(path: Path) -> dict[int, str] | None:
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    names = data.get("names")
    if isinstance(names, list):
        return {int(i): str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return None


def parse_yolo_labels(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
            rows.append((cid, x, y, w, h))
        except ValueError:
            continue
    return rows


def convert_coco_to_yolo(coco_json: Path, output_dir: Path) -> dict[str, Any]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
    ann_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        ann_by_image[str(img_id)].append(ann)

    images = {int(img.get("id")): img for img in data.get("images", []) if img.get("id") is not None}
    converted = 0
    reports: dict[str, Any] = {"converted_files": 0, "images_processed": 0, "mismatched_images": 0}
    for image_id, info in images.items():
        image_path = Path(info.get("file_name", ""))
        if not image_path:
            continue
        labels = ann_by_image.get(str(image_id), [])
        if not labels:
            continue
        target_dir = output_dir / image_path.parent.parent if image_path.parent.name.lower() in {"images", "train", "val"} else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        txt_path = target_dir / (image_path.stem + ".txt")
        rows: list[str] = []
        for ann in labels:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in categories:
                continue
            rows.append(f"{cat_id} {x} {y} {w} {h}")
        if rows:
            txt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            converted += 1
    reports["converted_files"] = converted
    reports["images_processed"] = len(images)
    return reports


def parse_voc_bboxes(xml_path: Path) -> list[tuple[str, float, float, float, float]]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    items: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall(".//object"):
        name = obj.findtext("name", default="")
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except ValueError:
            continue
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        items.append((name, xmin, ymin, w, h))
    return items


def convert_voc_to_yolo(voc_dir: Path, image_dir: Path, output_dir: Path) -> dict[str, Any]:
    xml_files = sorted(voc_dir.rglob("*.xml"))
    converted = 0
    for xml_path in xml_files:
        image_name = xml_path.stem
        image_candidate = image_dir / (image_name + ".jpg")
        if not image_candidate.exists():
            image_candidate = image_dir / (image_name + ".png")
        if not image_candidate.exists():
            continue
        rows = parse_voc_bboxes(xml_path)
        if not rows:
            continue
        out_path = output_dir / (image_name + ".txt")
        converted_rows: list[str] = []
        for name, xmin, ymin, w, h in rows:
            class_id = 0
            for idx, expected in EXPECTED_CLASSES.items():
                if expected.lower() == name.lower():
                    class_id = idx
                    break
            converted_rows.append(f"{class_id} {xmin} {ymin} {w} {h}")
        if converted_rows:
            out_path.write_text("\n".join(converted_rows) + "\n", encoding="utf-8")
            converted += 1
    return {"converted_files": converted, "xml_files": len(xml_files)}


def make_pipeline_report(dataset_root: Path, report_path: Path) -> dict[str, Any]:
    image_files = count_images(dataset_root)
    annotation_search = search_for_annotations(ROOT)
    txt_files = find_files(ROOT, {".txt"})
    json_files = find_files(ROOT, {".json"})
    xml_files = find_files(ROOT, {".xml"})
    yaml_files = find_files(ROOT, {".yaml", ".yml"})
    report = {
        "status": "BLOCKED",
        "dataset": str(dataset_root),
        "total_images": len(image_files),
        "total_annotation_files": len(txt_files) + len(json_files) + len(xml_files),
        "image_extensions": sorted({p.suffix.lower() for p in image_files}),
        "annotation_files": {
            "txt": [str(p.relative_to(ROOT)) for p in txt_files],
            "json": [str(p.relative_to(ROOT)) for p in json_files],
            "xml": [str(p.relative_to(ROOT)) for p in xml_files],
            "yaml": [str(p.relative_to(ROOT)) for p in yaml_files],
        },
        "annotations_found": False,
        "blocker": "The available images do not contain ground-truth defect annotations.",
        "expected_classes": EXPECTED_CLASSES,
        "search_summary": annotation_search,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="Dataset root to inspect")
    parser.add_argument("--report", type=Path, default=Path("reports/pipeline_readiness.json"), help="Pipeline readiness report path")
    args = parser.parse_args()

    dataset_root = (ROOT / args.root).resolve() if not args.root.is_absolute() else args.root
    report = make_pipeline_report(dataset_root, args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
