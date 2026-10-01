import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-workers", type=int, default=8, help="Workers for label sync.")
    parser.add_argument("--copy-labels", action=argparse.BooleanOptionalAction, default=False, help="Force copying labels instead of hardlinking when possible.")
    parser.add_argument("--skip-label-sync", action=argparse.BooleanOptionalAction, default=False, help="Skip labels_yolo -> labels synchronization.")
    return parser.parse_args()


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
    print("weights:", args.weights)
    print("epochs:", args.epochs)
    print("imgsz:", args.imgsz)
    print("batch:", args.batch)
    print("device:", args.device)
    print("project:", args.project)
    print("name:", args.name)
    if args.resume_from:
        print("resume_from:", args.resume_from)
        print("amp:", args.amp)

    if args.resume_from:
        model = YOLO(args.resume_from)
        results = model.train(
            resume=True,
            device=args.device,
            workers=args.workers,
            batch=args.batch,
            amp=args.amp,
        )
    else:
        model = YOLO(args.weights)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            amp=args.amp,
        )

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


def sync_ultralytics_labels(data_yaml_path: str, workers: int = 8, force_copy: bool = False) -> None:
    data_path = Path(data_yaml_path)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))

    base_path = Path(config.get("path", "."))
    if not base_path.is_absolute():
        cwd_candidate = (Path.cwd() / base_path).resolve()
        yaml_candidate = (data_path.parent / base_path).resolve()
        if cwd_candidate.exists():
            base_path = cwd_candidate
        else:
            base_path = yaml_candidate

    for split_key in ("train", "val"):
        split_images_rel = config.get(split_key)
        if not split_images_rel:
            continue
        split_images_dir = (base_path / split_images_rel).resolve()
        split_root = split_images_dir.parent
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
