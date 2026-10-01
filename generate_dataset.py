from __future__ import annotations

import argparse
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import DatasetConfig
from utils.exporters import (
    ensure_split_dirs,
    save_classes_json,
    save_data_yaml,
    save_image,
    save_json_annotation,
    save_yolo_annotation,
)
from utils.image_ops import (
    GradientSpec,
    ObjectAsset,
    apply_object_transforms,
    build_class_map,
    choose_scene_mode,
    choose_image_gradient,
    colorize_mask_with_duotone,
    colorize_mask_with_gradient,
    crop_to_alpha,
    get_tight_bbox_from_alpha,
    load_background_paths,
    load_object_assets,
    prepare_background,
)
from utils.placement import find_position_for_object
from utils.visualization import render_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic YOLO dataset generator.")
    parser.add_argument("--objects", type=str, default="objects", help="Path to object PNG files.")
    parser.add_argument("--backgrounds", type=str, default="backgrounds", help="Path to background images.")
    parser.add_argument("--output", type=str, default="output", help="Output directory.")
    parser.add_argument("--count", type=int, default=5000, help="Number of images to generate.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio [0,1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--preview", action="store_true", help="Save preview images with bbox.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview saving.")
    parser.add_argument("--min-objects", type=int, default=None, help="Minimum objects per image.")
    parser.add_argument("--max-objects", type=int, default=None, help="Maximum objects per image.")
    parser.add_argument("--balance-classes", action="store_true", help="Enable class balancing.")
    parser.add_argument("--no-balance-classes", action="store_true", help="Disable class balancing.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: generate a small dataset quickly.")
    parser.add_argument(
        "--bg-profile",
        choices=["soft", "aggressive", "small_moderate", "small_aggressive", "small_stretch_extreme"],
        default="soft",
        help="Generation profile (background + object size/shape behavior).",
    )
    parser.add_argument("--visual-test-count", type=int, default=None, help="Override count for visual test generation.")
    parser.add_argument("--small-object-bias", type=float, default=None, help="Bias to sample smaller scales [0..1].")
    parser.add_argument("--tiny-scale-prob", type=float, default=None, help="Probability of extra tiny object scale [0..1].")
    parser.add_argument("--scale-min", type=float, default=None, help="Minimum object scale.")
    parser.add_argument("--scale-max", type=float, default=None, help="Maximum object scale.")
    parser.add_argument("--stretch-x-min", type=float, default=None, help="Minimum stretch X.")
    parser.add_argument("--stretch-x-max", type=float, default=None, help="Maximum stretch X.")
    parser.add_argument("--stretch-y-min", type=float, default=None, help="Minimum stretch Y.")
    parser.add_argument("--stretch-y-max", type=float, default=None, help="Maximum stretch Y.")
    parser.add_argument("--anisotropic-prob", type=float, default=None, help="Probability of anisotropic stretch [0..1].")
    parser.add_argument("--mono-black-bg-prob", type=float, default=None, help="Probability of black background + white objects [0..1].")
    parser.add_argument("--mono-white-bg-prob", type=float, default=None, help="Probability of white background + black objects [0..1].")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of processes for generation.")
    parser.add_argument("--worker-chunk-size", type=int, default=64, help="Samples per worker task in multiprocess mode.")
    return parser.parse_args()


def choose_split_indices(count: int, train_split: float, rng: random.Random) -> Dict[int, str]:
    idx = list(range(count))
    rng.shuffle(idx)
    train_count = int(round(count * train_split))
    train_set = set(idx[:train_count])
    return {i: ("train" if i in train_set else "val") for i in range(count)}


def sample_assets_for_image(
    assets: List[ObjectAsset],
    amount: int,
    class_usage: Dict[str, int],
    rng: random.Random,
    balance_classes: bool,
) -> List[ObjectAsset]:
    class_to_assets: Dict[str, List[ObjectAsset]] = {}
    for asset in assets:
        class_to_assets.setdefault(asset.class_name, []).append(asset)

    available_classes = list(class_to_assets.keys())
    amount = min(amount, len(available_classes))
    if amount <= 0:
        return []

    if not balance_classes:
        chosen_classes = rng.sample(available_classes, k=amount)
    else:
        # Unique-class weighted sampling: pick classes one-by-one without replacement.
        chosen_classes: List[str] = []
        pool = available_classes[:]
        for _ in range(amount):
            weights = [1.0 / (1 + class_usage[c]) for c in pool]
            picked = rng.choices(pool, weights=weights, k=1)[0]
            chosen_classes.append(picked)
            pool.remove(picked)

    return [rng.choice(class_to_assets[c]) for c in chosen_classes]


def generate_one_image(
    sample_idx: int,
    cfg: DatasetConfig,
    assets: List[ObjectAsset],
    background_paths: List[Path],
    class_usage: Dict[str, int],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[Image.Image, List[Dict]]:
    unique_classes_count = len({a.class_name for a in assets})
    if unique_classes_count <= 0:
        raise RuntimeError("No unique classes available for generation.")

    effective_max = min(cfg.max_objects_per_image, unique_classes_count)
    effective_min = min(cfg.min_objects_per_image, effective_max)

    for _ in range(cfg.max_scene_attempts):
        scene_mode = choose_scene_mode(cfg, rng)
        if scene_mode == "mono_black_bg":
            canvas = Image.new("RGB", (cfg.image_width, cfg.image_height), (0, 0, 0))
        elif scene_mode == "mono_white_bg":
            canvas = Image.new("RGB", (cfg.image_width, cfg.image_height), (255, 255, 255))
        else:
            bg_path = rng.choice(background_paths)
            canvas = prepare_background(bg_path, cfg=cfg, rng=rng, np_rng=np_rng)
        canvas_rgba = canvas.convert("RGBA")

        target_objects = rng.randint(effective_min, effective_max)
        selected_assets = sample_assets_for_image(
            assets=assets,
            amount=target_objects,
            class_usage=class_usage,
            rng=rng,
            balance_classes=cfg.balance_classes,
        )
        gradient: GradientSpec = choose_image_gradient(cfg, rng)

        placed_bboxes: List[Tuple[int, int, int, int]] = []
        objects_info: List[Dict] = []

        for asset in selected_assets:
            if scene_mode == "mono_black_bg":
                obj_rgba = colorize_mask_with_duotone(
                    asset.mask,
                    c1=(220, 220, 220),
                    c2=(255, 255, 255),
                    angle=rng.uniform(0, np.pi * 2.0),
                )
                apply_appearance = False
            elif scene_mode == "mono_white_bg":
                obj_rgba = colorize_mask_with_duotone(
                    asset.mask,
                    c1=(0, 0, 0),
                    c2=(46, 46, 46),
                    angle=rng.uniform(0, np.pi * 2.0),
                )
                apply_appearance = False
            else:
                obj_rgba = colorize_mask_with_gradient(asset.mask, gradient=gradient)
                apply_appearance = True
            obj_rgba = apply_object_transforms(
                obj_rgba,
                cfg=cfg,
                rng=rng,
                np_rng=np_rng,
                apply_appearance=apply_appearance,
                allow_flip=asset.class_name not in cfg.no_flip_classes,
            )
            obj_rgba = crop_to_alpha(obj_rgba)
            if obj_rgba is None:
                continue

            if obj_rgba.width < cfg.min_object_size_px or obj_rgba.height < cfg.min_object_size_px:
                continue

            # The crop keeps the faint anti-aliased fringe for rendering; the label uses
            # the visible object only (see DatasetConfig.bbox_alpha_threshold).
            tight = get_tight_bbox_from_alpha(obj_rgba, threshold=cfg.bbox_alpha_threshold)
            if tight is None:
                continue

            pos = find_position_for_object(
                obj_w=obj_rgba.width,
                obj_h=obj_rgba.height,
                placed_bboxes=placed_bboxes,
                image_width=cfg.image_width,
                image_height=cfg.image_height,
                max_attempts=cfg.max_placement_attempts,
                edge_padding=cfg.edge_padding_px,
                min_distance=cfg.min_distance_between_boxes,
                rng=rng,
            )
            if pos is None:
                continue

            x, y = pos
            canvas_rgba.alpha_composite(obj_rgba, dest=(x, y))

            bbox = {
                "x_min": x + tight[0],
                "y_min": y + tight[1],
                "x_max": x + tight[2],
                "y_max": y + tight[3],
            }
            # Collision checks use the full pasted area so fringes of neighbours never overlap.
            placed_bboxes.append((x, y, x + obj_rgba.width - 1, y + obj_rgba.height - 1))
            objects_info.append(
                {
                    "class": asset.class_name,
                    "class_id": asset.class_id,
                    "bbox": bbox,
                }
            )

        if len(objects_info) >= effective_min:
            for obj in objects_info:
                class_usage[obj["class"]] += 1
            return canvas_rgba.convert("RGB"), objects_info

    raise RuntimeError(
        f"Could not place enough objects for sample {sample_idx}. "
        f"Needed at least {effective_min} unique objects."
    )


def build_annotation(image_name: str, cfg: DatasetConfig, objects_info: List[Dict]) -> Dict:
    return {
        "image": image_name,
        "width": cfg.image_width,
        "height": cfg.image_height,
        "objects": objects_info,
    }


def build_config_from_args(args: argparse.Namespace) -> DatasetConfig:
    cfg = DatasetConfig.with_profile(args.bg_profile)
    cfg = replace(
        cfg,
        dataset_size=args.count,
        train_split=args.train_split,
        random_seed=args.seed,
    )
    if args.min_objects is not None:
        cfg.min_objects_per_image = max(1, args.min_objects)
    if args.max_objects is not None:
        cfg.max_objects_per_image = max(cfg.min_objects_per_image, args.max_objects)
    if args.preview:
        cfg.save_preview = True
    if args.no_preview:
        cfg.save_preview = False
    if args.balance_classes:
        cfg.balance_classes = True
    if args.no_balance_classes:
        cfg.balance_classes = False
    if args.debug:
        cfg.debug_mode = True
        cfg.dataset_size = cfg.debug_dataset_size
    if args.visual_test_count is not None and args.visual_test_count > 0:
        cfg.visual_test_count = args.visual_test_count
        cfg.dataset_size = args.visual_test_count
    if args.small_object_bias is not None:
        cfg.small_object_bias = float(np.clip(args.small_object_bias, 0.0, 1.0))
    if args.tiny_scale_prob is not None:
        cfg.tiny_scale_prob = float(np.clip(args.tiny_scale_prob, 0.0, 1.0))
    if args.scale_min is not None or args.scale_max is not None:
        s_min, s_max = cfg.scale_range
        s_min = args.scale_min if args.scale_min is not None else s_min
        s_max = args.scale_max if args.scale_max is not None else s_max
        cfg.scale_range = (float(min(s_min, s_max)), float(max(s_min, s_max)))
    if args.stretch_x_min is not None or args.stretch_x_max is not None:
        x_min, x_max = cfg.stretch_x_range
        x_min = args.stretch_x_min if args.stretch_x_min is not None else x_min
        x_max = args.stretch_x_max if args.stretch_x_max is not None else x_max
        cfg.stretch_x_range = (float(min(x_min, x_max)), float(max(x_min, x_max)))
    if args.stretch_y_min is not None or args.stretch_y_max is not None:
        y_min, y_max = cfg.stretch_y_range
        y_min = args.stretch_y_min if args.stretch_y_min is not None else y_min
        y_max = args.stretch_y_max if args.stretch_y_max is not None else y_max
        cfg.stretch_y_range = (float(min(y_min, y_max)), float(max(y_min, y_max)))
    if args.anisotropic_prob is not None:
        cfg.anisotropic_stretch_prob = float(np.clip(args.anisotropic_prob, 0.0, 1.0))
    if args.mono_black_bg_prob is not None:
        cfg.mono_black_bg_prob = float(np.clip(args.mono_black_bg_prob, 0.0, 1.0))
    if args.mono_white_bg_prob is not None:
        cfg.mono_white_bg_prob = float(np.clip(args.mono_white_bg_prob, 0.0, 1.0))
    return cfg


def validate_inputs(objects_dir: Path, backgrounds_dir: Path, cfg: DatasetConfig) -> None:
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory does not exist: {objects_dir}")
    if not backgrounds_dir.exists():
        raise FileNotFoundError(f"Backgrounds directory does not exist: {backgrounds_dir}")
    if cfg.dataset_size < 1:
        raise ValueError("--count must be at least 1")
    if not (0.0 < cfg.train_split < 1.0):
        raise ValueError("--train-split must be in (0, 1)")
    if cfg.min_objects_per_image > cfg.max_objects_per_image:
        raise ValueError("min_objects_per_image cannot be greater than max_objects_per_image")


# Top-level entries a dataset produced by this script may contain.
DATASET_ENTRIES = {"train", "val", "data.yaml", "classes.json"}


def prepare_output_dir(output_dir: Path, protected_dirs: List[Path]) -> None:
    """Recreate output_dir, refusing to wipe anything that is not a previous dataset."""
    out = output_dir.resolve()
    for protected in protected_dirs:
        protected = protected.resolve()
        if out == protected or out in protected.parents:
            raise ValueError(f"Refusing to use '{output_dir}' as output: it is or contains '{protected}'.")

    if out.exists():
        if not out.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        unexpected = sorted(p.name for p in out.iterdir() if p.name not in DATASET_ENTRIES)
        if unexpected:
            raise ValueError(
                f"Output directory '{output_dir}' is not empty and does not look like a dataset "
                f"produced by this script (unexpected: {', '.join(unexpected[:5])}). "
                "Choose another --output or remove the directory manually."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)

    objects_dir = Path(args.objects)
    backgrounds_dir = Path(args.backgrounds)
    output_dir = Path(args.output)

    validate_inputs(objects_dir, backgrounds_dir, cfg)

    rng = random.Random(cfg.random_seed)
    np_rng = np.random.default_rng(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    object_paths = sorted(objects_dir.glob("*.png"))
    background_paths = load_background_paths(backgrounds_dir)
    if not object_paths:
        raise RuntimeError(f"No PNG files found in {objects_dir}")
    if not background_paths:
        raise RuntimeError(f"No PNG files found in {backgrounds_dir}")

    class_map = build_class_map(object_paths)
    assets = load_object_assets(objects_dir, class_map, cfg)
    if not assets:
        raise RuntimeError("No valid object assets available after mask extraction.")
    unique_classes_count = len({a.class_name for a in assets})
    if unique_classes_count < cfg.min_objects_per_image:
        print(
            "[WARNING] Unique objects are fewer than requested minimum: "
            f"{unique_classes_count} < {cfg.min_objects_per_image}. "
            f"Generator will use up to {unique_classes_count} unique objects per image."
        )

    prepare_output_dir(
        output_dir,
        protected_dirs=[Path.cwd(), Path(__file__).resolve().parent, objects_dir, backgrounds_dir],
    )

    split_map = choose_split_indices(cfg.dataset_size, cfg.train_split, rng)
    split_dirs = {
        "train": ensure_split_dirs(output_dir, "train", cfg.save_preview),
        "val": ensure_split_dirs(output_dir, "val", cfg.save_preview),
    }

    class_usage = {name: 0 for name in class_map.keys()}

    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for idx in tqdm(range(cfg.dataset_size), desc="Generating dataset"):
            image_name = f"{idx + 1:06d}.png"
            base_name = f"{idx + 1:06d}"
            split = split_map[idx]
            dirs = split_dirs[split]

            image, objects_info = generate_one_image(
                sample_idx=idx,
                cfg=cfg,
                assets=assets,
                background_paths=background_paths,
                class_usage=class_usage,
                rng=rng,
                np_rng=np_rng,
            )

            annotation = build_annotation(image_name=image_name, cfg=cfg, objects_info=objects_info)
            preview = render_preview(image, objects_info) if cfg.save_preview else None

            save_image(image, dirs["images"] / image_name)
            save_json_annotation(annotation, dirs["labels_json"] / f"{base_name}.json")
            save_yolo_annotation(
                objects=objects_info,
                image_width=cfg.image_width,
                image_height=cfg.image_height,
                out_path=dirs["labels_yolo"] / f"{base_name}.txt",
            )
            if cfg.save_preview and preview is not None:
                save_image(preview, dirs["preview"] / image_name)
    else:
        num_workers = min(num_workers, os.cpu_count() or num_workers, cfg.dataset_size)
        indices = list(range(cfg.dataset_size))
        chunk_size = max(1, int(args.worker_chunk_size))
        chunks = _split_indices_by_size(indices, chunk_size)
        worker_args = [
            (
                wid,
                chunk,
                cfg,
                objects_dir,
                backgrounds_dir,
                class_map,
                split_map,
                split_dirs,
            )
            for wid, chunk in enumerate(chunks)
            if chunk
        ]

        ex = ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn"))
        futures = [ex.submit(_generate_chunk, *wa) for wa in worker_args]
        interrupted = False
        try:
            with tqdm(total=cfg.dataset_size, desc=f"Generating dataset ({num_workers} workers)") as pbar:
                for fut in as_completed(futures):
                    processed, worker_usage = fut.result()
                    pbar.update(processed)
                    for k, v in worker_usage.items():
                        class_usage[k] += v
        except KeyboardInterrupt:
            interrupted = True
            print("\n[INFO] Interrupted by user. Cancelling pending worker tasks...")
            for fut in futures:
                fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
        finally:
            if not interrupted:
                ex.shutdown(wait=True, cancel_futures=False)
        if interrupted:
            raise SystemExit(130)

    save_classes_json(class_map, output_dir)
    save_data_yaml(class_map, output_dir)

    print("Done.")
    print(f"Output: {output_dir.resolve()}")
    print(f"Total images: {cfg.dataset_size}")
    print(f"Class usage: {class_usage}")


def _split_indices_by_size(indices: List[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        return [indices]
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def _generate_chunk(
    worker_id: int,
    indices: List[int],
    cfg: DatasetConfig,
    objects_dir: Path,
    backgrounds_dir: Path,
    class_map: Dict[str, int],
    split_map: Dict[int, str],
    split_dirs: Dict[str, Dict[str, Path]],
) -> Tuple[int, Dict[str, int]]:
    seed = cfg.random_seed + (worker_id + 1) * 1_000_003
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    assets = load_object_assets(objects_dir, class_map, cfg)
    background_paths = load_background_paths(backgrounds_dir)
    class_usage = {name: 0 for name in class_map.keys()}

    for idx in indices:
        image_name = f"{idx + 1:06d}.png"
        base_name = f"{idx + 1:06d}"
        split = split_map[idx]
        dirs = split_dirs[split]

        image, objects_info = generate_one_image(
            sample_idx=idx,
            cfg=cfg,
            assets=assets,
            background_paths=background_paths,
            class_usage=class_usage,
            rng=rng,
            np_rng=np_rng,
        )

        annotation = build_annotation(image_name=image_name, cfg=cfg, objects_info=objects_info)
        preview = render_preview(image, objects_info) if cfg.save_preview else None

        save_image(image, dirs["images"] / image_name)
        save_json_annotation(annotation, dirs["labels_json"] / f"{base_name}.json")
        save_yolo_annotation(
            objects=objects_info,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
            out_path=dirs["labels_yolo"] / f"{base_name}.txt",
        )
        if cfg.save_preview and preview is not None:
            save_image(preview, dirs["preview"] / image_name)

    return len(indices), class_usage


if __name__ == "__main__":
    main()
