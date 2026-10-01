from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(
    *,
    image_info: dict[str, Any],
    model_path: Path,
    detections: list[dict[str, Any]],
    inference_ms: float,
    conf_thres: float,
    iou_thres: float,
    device: str,
) -> dict[str, Any]:
    unique_classes = sorted({d["class_name"] for d in detections})
    avg_conf = round(sum(d["confidence"] for d in detections) / len(detections), 4) if detections else 0.0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "device": device,
        "thresholds": {"confidence": conf_thres, "iou": iou_thres},
        "image": image_info,
        "summary": {
            "total_objects": len(detections),
            "unique_classes": unique_classes,
            "avg_confidence": avg_conf,
            "inference_ms": round(inference_ms, 2),
        },
        "detections": detections,
    }


def save_report_json(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_report_csv(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "area",
        "center_x",
        "center_y",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in report.get("detections", []):
            b = d["bbox"]
            c = d["center"]
            writer.writerow(
                {
                    "index": d["index"],
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "confidence": f'{d["confidence"]:.6f}',
                    "x_min": b["x_min"],
                    "y_min": b["y_min"],
                    "x_max": b["x_max"],
                    "y_max": b["y_max"],
                    "width": d["width"],
                    "height": d["height"],
                    "area": d["area"],
                    "center_x": c["x"],
                    "center_y": c["y"],
                }
            )

