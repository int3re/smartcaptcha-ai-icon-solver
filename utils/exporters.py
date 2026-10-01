from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image


def ensure_split_dirs(base_output: Path, split_name: str, save_preview: bool) -> Dict[str, Path]:
    split_root = base_output / split_name
    dirs = {
        "root": split_root,
        "images": split_root / "images",
        "labels_json": split_root / "labels_json",
        "labels_yolo": split_root / "labels_yolo",
    }
    if save_preview:
        dirs["preview"] = split_root / "preview"
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_image(image: Image.Image, out_path: Path) -> None:
    image.save(out_path, format="PNG")


def save_json_annotation(annotation: Dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def save_yolo_annotation(
    objects: List[Dict],
    image_width: int,
    image_height: int,
    out_path: Path,
) -> None:
    lines = []
    for obj in objects:
        x_min = obj["bbox"]["x_min"]
        y_min = obj["bbox"]["y_min"]
        x_max = obj["bbox"]["x_max"]
        y_max = obj["bbox"]["y_max"]
        class_id = obj["class_id"]

        w = x_max - x_min + 1
        h = y_max - y_min + 1
        x_c = x_min + w / 2.0
        y_c = y_min + h / 2.0

        x_c_n = x_c / image_width
        y_c_n = y_c / image_height
        w_n = w / image_width
        h_n = h / image_height

        lines.append(f"{class_id} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f}")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_classes_json(class_map: Dict[str, int], output_dir: Path) -> None:
    with (output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)


def save_data_yaml(class_map: Dict[str, int], output_dir: Path, dataset_root_name: str) -> None:
    inverse_names = {class_id: class_name for class_name, class_id in class_map.items()}
    payload = {
        "path": dataset_root_name,
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_map),
        "names": inverse_names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
