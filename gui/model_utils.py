from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def find_model_candidates(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit = [
        project_root / "weights" / "yolov8n_gpu_best.pt",
        project_root / "weights" / "yolov8n_gpu_last.pt",
        project_root / "weights" / "yolov8n.pt",
        project_root / "yolov8n.pt",
    ]
    for p in explicit:
        if p.exists():
            candidates.append(p)

    runs_train = project_root / "runs" / "train"
    if runs_train.exists():
        run_dirs = sorted([d for d in runs_train.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            if best.exists():
                candidates.append(best)
            if last.exists():
                candidates.append(last)

    # Remove duplicates while preserving priority.
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    return uniq


def auto_detect_model(project_root: Path) -> Path | None:
    candidates = find_model_candidates(project_root)
    return candidates[0] if candidates else None


def load_yolo_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def run_inference(
    model: YOLO,
    image_path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "auto",
) -> dict[str, Any]:
    kwargs = {
        "source": str(image_path),
        "conf": conf_thres,
        "iou": iou_thres,
        "verbose": False,
    }
    if device and device != "auto":
        kwargs["device"] = device

    t0 = time.perf_counter()
    results = model.predict(**kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not results:
        return {"detections": [], "elapsed_ms": elapsed_ms, "raw_result": None}

    result = results[0]
    names = result.names
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            w = x2 - x1
            h = y2 - y1
            detections.append(
                {
                    "index": idx + 1,
                    "class_id": cls_id,
                    "class_name": str(names.get(cls_id, f"class_{cls_id}")),
                    "confidence": conf,
                    "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "center": {"x": int(round((x1 + x2) / 2)), "y": int(round((y1 + y2) / 2))},
                }
            )

    return {"detections": detections, "elapsed_ms": elapsed_ms, "raw_result": result}
