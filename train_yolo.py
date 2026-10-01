import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="output/data.yaml")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt")
    parser.add_argument("--resume-from", type=str, default="", help="Path to last.pt for resume training.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mixed precision (default: on). Ignored with --resume-from: Ultralytics restores it from the checkpoint.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="SGD, AdamW, ... With 'auto' Ultralytics picks optimizer and lr0 itself and ignores --lr0.",
    )
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (requires explicit --optimizer).")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs (Ultralytics default: 3). Use 0 for fine-tuning.")
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.0,
        help="Horizontal flip probability of the whole mosaic. Default 0: the generator already flips "
        "objects per class and must keep letter-like classes (DatasetConfig.no_flip_classes) unmirrored.",
    )
    parser.add_argument("--sync-workers", type=int, default=8, help="Workers for label sync.")
    parser.add_argument("--copy-labels", action=argparse.BooleanOptionalAction, default=False, help="Force copying labels instead of hardlinking when possible.")
    parser.add_argument("--skip-label-sync", action=argparse.BooleanOptionalAction, default=False, help="Skip labels_yolo -> labels synchronization.")
    args = parser.parse_args(argv)
    if args.lr0 is not None and args.optimizer.lower() == "auto":
        parser.error("--lr0 is ignored by Ultralytics with --optimizer auto; pass e.g. --optimizer SGD or AdamW.")
    return args


def main():
    args = parse_args()
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (Path.cwd() / project_path).resolve()
    args.project = str(project_path)

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    if args.resume_from and not Path(args.resume_from).exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")

    if not args.resume_from and not Path(args.weights).exists():
        fallback = Path("yolov8n.pt")
        if fallback.exists():
            print(f"WARNING: {args.weights} not found, using {fallback}")
            args.weights = str(fallback)
        else:
            raise FileNotFoundError(f"Weights not found: {args.weights}. Run prepare_yolo_model.py first.")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, switching to CPU")
        args.device = "cpu"

    if not args.skip_label_sync:
        sync_ultralytics_labels(
            data_yaml_path=args.data,
            workers=max(1, args.sync_workers),
            force_copy=args.copy_labels,
        )
    else:
        print("Label sync skipped by flag: --skip-label-sync")

    print("=== TRAIN CONFIG ===")
    print("data:", args.data)
    print("device:", args.device)
    print("batch:", args.batch)
    print("workers:", args.workers)

    if args.resume_from:
        # Ultralytics restores every hyperparameter from the checkpoint and only lets a few
        # memory/device related ones (batch, device, workers, ...) be overridden on resume.
        print("resume_from:", args.resume_from)
        if args.amp is not None:
            print("WARNING: --amp/--no-amp is ignored on resume; AMP setting is taken from the checkpoint.")
        model = YOLO(args.resume_from)
        results = model.train(
            resume=True,
            device=args.device,
            workers=args.workers,
            batch=args.batch,
        )
    else:
        train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            amp=True if args.amp is None else args.amp,
            optimizer=args.optimizer,
            fliplr=args.fliplr,
        )
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        if args.warmup_epochs is not None:
            train_kwargs["warmup_epochs"] = args.warmup_epochs
        for key in ("weights", "epochs", "imgsz", "project", "name"):
            print(f"{key}:", getattr(args, key))
        for key in ("amp", "optimizer", "lr0", "warmup_epochs", "fliplr"):
            print(f"{key}:", train_kwargs.get(key, "(ultralytics default)"))
        model = YOLO(args.weights)
        results = model.train(**train_kwargs)

    print("\nTraining finished")
    print("Results saved to:", results.save_dir)


def _link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_dataset_root(data_path: Path, config: dict) -> Path:
    """Dataset root the same way Ultralytics sees it (for datasets kept inside the project)."""
    raw = config.get("path")
    if not raw:
        # Without "path" Ultralytics resolves splits relative to the yaml file.
        return data_path.parent.resolve()
    base_path = Path(raw)
    if base_path.is_absolute():
        return base_path
    cwd_candidate = (Path.cwd() / base_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (data_path.parent / base_path).resolve()


def sync_ultralytics_labels(data_yaml_path: str, workers: int = 8, force_copy: bool = False) -> None:
    data_path = Path(data_yaml_path)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    base_path = resolve_dataset_root(data_path, config)

    synced_dirs: set[Path] = set()
    for split_key in ("train", "val", "test"):
        split_images_rel = config.get(split_key)
        if not split_images_rel:
            continue
        split_images_dir = (base_path / split_images_rel).resolve()
        split_root = split_images_dir.parent
        if split_root in synced_dirs:
            continue
        synced_dirs.add(split_root)
        yolo_labels_dir = split_root / "labels_yolo"
        labels_dir = split_root / "labels"
        if not yolo_labels_dir.exists():
            continue

        labels_dir.mkdir(parents=True, exist_ok=True)
        txt_files = list(yolo_labels_dir.glob("*.txt"))
        if not txt_files:
            print(f"[sync] {split_key}: no label files in {yolo_labels_dir}")
            continue

        missing: list[tuple[Path, Path]] = []
        for src in txt_files:
            dst = labels_dir / src.name
            if not dst.exists():
                missing.append((src, dst))

        print(
            f"[sync] {split_key}: total={len(txt_files)} existing={len(txt_files)-len(missing)} missing={len(missing)} "
            f"mode={'copy' if force_copy else 'hardlink->copy fallback'}"
        )
        if not missing:
            continue

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_link_or_copy, src, dst, force_copy) for src, dst in missing]
            with tqdm(total=len(futures), desc=f"sync {split_key}", unit="file") as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)


if __name__ == "__main__":
    main()
