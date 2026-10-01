"""Evaluate a trained model on a labelled dataset (e.g. a hand-labelled set of real images).

The synthetic val split comes from the same generator as the training data, so its
metrics say little about real images. Point --data at a separate test set to measure that.
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from train_yolo import sync_ultralytics_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a YOLO model on a dataset split and save metrics.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt model.")
    parser.add_argument("--data", type=str, required=True, help="data.yaml of the evaluation dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=None, help="Default: image size the model was trained with.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for mAP computation.")
    parser.add_argument("--project", type=str, default="runs/val", help="Directory for validation runs (plots, metrics).")
    parser.add_argument("--name", type=str, default="", help="Run name (default: <model run>_<dataset>).")
    parser.add_argument("--out", type=str, default="", help="Where to write metrics JSON (default: <run dir>/metrics.json).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    if args.device != "cpu" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, switching to CPU")
        args.device = "cpu"

    sync_ultralytics_labels(args.data)

    weights_path, data_path = Path(args.weights).resolve(), Path(args.data).resolve()
    name = args.name or f"{weights_path.parent.parent.name}_{data_path.parent.name}_{args.split}"
    project = Path(args.project)
    if not project.is_absolute():
        project = (Path.cwd() / project).resolve()

    model = YOLO(args.weights)
    val_kwargs = dict(
        data=args.data,
        split=args.split,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        plots=True,
        project=str(project),
        name=name,
    )
    if args.imgsz:
        val_kwargs["imgsz"] = args.imgsz
    metrics = model.val(**val_kwargs)

    box = metrics.box
    names = metrics.names
    per_class = {names[int(c)]: round(float(m), 5) for c, m in zip(metrics.ap_class_index, box.maps[metrics.ap_class_index])}
    report = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
        "mAP50": round(float(box.map50), 5),
        "mAP50-95": round(float(box.map), 5),
        "per_class_mAP50-95": per_class,
        "worst_classes": sorted(per_class.items(), key=lambda kv: kv[1])[:10],
    }

    out_path = Path(args.out) if args.out else Path(metrics.save_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"precision={report['precision']} recall={report['recall']} mAP50={report['mAP50']} mAP50-95={report['mAP50-95']}")
    print("worst classes:", ", ".join(f"{n}={v}" for n, v in report["worst_classes"]))
    print("Metrics saved to:", out_path)


if __name__ == "__main__":
    main()
