from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# COCO-pretrained starting checkpoints: they do not know the icon classes.
BASE_CHECKPOINTS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo26n.pt"}
SCORE_COLUMN = "metrics/mAP50-95(B)"


def run_best_score(run_dir: Path) -> float | None:
    """Best validation mAP50-95 recorded in an Ultralytics run, if results.csv is present."""
    results_csv = run_dir / "results.csv"
    try:
        with results_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    scores = []
    for row in rows:
        row = {k.strip(): v for k, v in row.items() if k}
        try:
            scores.append(float(row[SCORE_COLUMN]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(scores) if scores else None


def find_model_candidates(project_root: Path) -> list[Path]:
    """Trained models, best first.

    Runs in runs/train are ranked by their best mAP50-95 (runs without results.csv go
    last, newest first). Scores are only comparable between runs validated on the same
    val set; the user can always pick another model manually.
    """
    candidates: list[Path] = []

    runs_train = project_root / "runs" / "train"
    if runs_train.exists():
        scores = {d: run_best_score(d) for d in runs_train.iterdir() if d.is_dir()}
        run_dirs = sorted(
            scores,
            key=lambda d: (scores[d] is not None, scores[d] or 0.0, d.stat().st_mtime),
            reverse=True,
        )
        for run_dir in run_dirs:
            candidates.extend(p for p in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt") if p.exists())

    weights_dir = project_root / "weights"
    if weights_dir.exists():
        trained = [p for p in weights_dir.glob("*.pt") if p.name not in BASE_CHECKPOINTS]
        trained.sort(key=lambda p: ("best" not in p.stem, -p.stat().st_mtime))
        candidates.extend(trained)

    # Remove duplicates while preserving priority.
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    return uniq


def auto_detect_model(project_root: Path, preferred: Path | str | None = None) -> Path | None:
    """The model the user picked last time if it still exists, otherwise the best candidate."""
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.is_file():
            return preferred_path.resolve()
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
