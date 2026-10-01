from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import DatasetConfig


@dataclass
class ObjectAsset:
    class_name: str
    class_id: int
    path: Path
    mask: Image.Image  # mode "L"


@dataclass(frozen=True)
class GradientSpec:
    c1: Tuple[int, int, int]
    c2: Tuple[int, int, int]
    kind: str  # "linear" | "radial" | "hard_split" | "banded"
    angle: float
    center_x: float
    center_y: float


def _extract_class_name(path: Path) -> str:
    return path.stem


def build_class_map(object_paths: List[Path]) -> Dict[str, int]:
    class_names = sorted({_extract_class_name(p) for p in object_paths})
    return {name: idx for idx, name in enumerate(class_names)}


def load_object_assets(objects_dir: Path, class_map: Dict[str, int], cfg: DatasetConfig) -> List[ObjectAsset]:
    assets: List[ObjectAsset] = []
    for path in sorted(objects_dir.glob("*.png")):
        img = Image.open(path).convert("RGB")
        gray = np.array(img.convert("L"))
        # White-ish background becomes transparent; darker pixels are object.
        mask = np.where(gray < cfg.white_bg_threshold, 255, 0).astype(np.uint8)
        if mask.max() == 0:
            continue
        class_name = _extract_class_name(path)
        assets.append(
            ObjectAsset(
                class_name=class_name,
                class_id=class_map[class_name],
                path=path,
                mask=Image.fromarray(mask, mode="L"),
            )
        )
    return assets


def load_background_paths(backgrounds_dir: Path) -> List[Path]:
    return sorted(backgrounds_dir.glob("*.png"))


def _random_color(rng: random.Random) -> Tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def _sample_gradient_palette(cfg: DatasetConfig, rng: random.Random) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if not cfg.gradient_palettes:
        return _random_color(rng), _random_color(rng)
    return rng.choice(cfg.gradient_palettes)


def choose_image_gradient(cfg: DatasetConfig, rng: random.Random) -> GradientSpec:
    c1, c2 = _sample_gradient_palette(cfg, rng)
    modes = [
        ("linear", max(0.0, cfg.gradient_linear_prob)),
        ("radial", max(0.0, cfg.gradient_radial_prob)),
        ("hard_split", max(0.0, cfg.gradient_hard_split_prob)),
        ("banded", max(0.0, cfg.gradient_banded_prob)),
    ]
    kinds = [m[0] for m in modes]
    weights = [m[1] for m in modes]
    if sum(weights) <= 0:
        kind = "hard_split"
    else:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
    return GradientSpec(
        c1=c1,
        c2=c2,
        kind=kind,
        angle=rng.uniform(0, math.pi * 2.0),
        center_x=rng.uniform(0.2, 0.8),
        center_y=rng.uniform(0.2, 0.8),
    )


def choose_scene_mode(cfg: DatasetConfig, rng: random.Random) -> str:
    p_black = max(0.0, cfg.mono_black_bg_prob)
    p_white = max(0.0, cfg.mono_white_bg_prob)
    total = p_black + p_white
    if total >= 1.0:
        p_black = p_black / total
        p_white = p_white / total
    r = rng.random()
    if r < p_black:
        return "mono_black_bg"
    if r < p_black + p_white:
        return "mono_white_bg"
    return "normal"


def _make_linear_gradient(width: int, height: int, c1: Tuple[int, int, int], c2: Tuple[int, int, int], angle: float) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    proj = (xx * math.cos(angle) + yy * math.sin(angle))
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    c1_arr = np.array(c1, dtype=np.float32)
    c2_arr = np.array(c2, dtype=np.float32)
    grad = c1_arr[None, None, :] * (1.0 - t[:, :, None]) + c2_arr[None, None, :] * t[:, :, None]
    return grad.astype(np.uint8)


def _make_hard_split_gradient(
    width: int,
    height: int,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    angle: float,
    alpha_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    if alpha_mask is not None and np.any(alpha_mask):
        # Split by object pixels only: closest to true 50/50 color split over the symbol itself.
        threshold = float(np.median(proj[alpha_mask > 0]))
    else:
        threshold = float(np.median(proj))
    selector = proj >= threshold
    grad = np.empty((height, width, 3), dtype=np.uint8)
    grad[:, :, 0] = np.where(selector, c2[0], c1[0])
    grad[:, :, 1] = np.where(selector, c2[1], c1[1])
    grad[:, :, 2] = np.where(selector, c2[2], c1[2])
    return grad


def _make_banded_gradient(
    width: int,
    height: int,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    angle: float,
    bands: int,
) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    steps = np.floor(t * max(2, bands)).astype(np.int32)
    selector = (steps % 2) == 0
    grad = np.empty((height, width, 3), dtype=np.uint8)
    grad[:, :, 0] = np.where(selector, c1[0], c2[0])
    grad[:, :, 1] = np.where(selector, c1[1], c2[1])
    grad[:, :, 2] = np.where(selector, c1[2], c2[2])
    return grad


def _make_radial_gradient(
    width: int,
    height: int,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    center_x: float,
    center_y: float,
) -> np.ndarray:
    cx = center_x * width
    cy = center_y * height
    yy, xx = np.indices((height, width))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(dist / (dist.max() + 1e-8), 0.0, 1.0)
    c1_arr = np.array(c1, dtype=np.float32)
    c2_arr = np.array(c2, dtype=np.float32)
    grad = c1_arr[None, None, :] * (1.0 - t[:, :, None]) + c2_arr[None, None, :] * t[:, :, None]
    return grad.astype(np.uint8)


def colorize_mask_with_gradient(mask: Image.Image, gradient: GradientSpec) -> Image.Image:
    width, height = mask.size
    alpha = np.array(mask, dtype=np.uint8)
    if gradient.kind == "linear":
        grad_rgb = _make_linear_gradient(width, height, gradient.c1, gradient.c2, gradient.angle)
    elif gradient.kind == "hard_split":
        grad_rgb = _make_hard_split_gradient(
            width=width,
            height=height,
            c1=gradient.c1,
            c2=gradient.c2,
            angle=gradient.angle,
            alpha_mask=alpha,
        )
    elif gradient.kind == "banded":
        grad_rgb = _make_banded_gradient(
            width=width,
            height=height,
            c1=gradient.c1,
            c2=gradient.c2,
            angle=gradient.angle,
            bands=max(3, int(round(4 + gradient.center_y * 4))),
        )
    else:
        grad_rgb = _make_radial_gradient(
            width=width,
            height=height,
            c1=gradient.c1,
            c2=gradient.c2,
            center_x=gradient.center_x,
            center_y=gradient.center_y,
        )

    rgba = np.dstack([grad_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def colorize_mask_with_duotone(mask: Image.Image, c1: Tuple[int, int, int], c2: Tuple[int, int, int], angle: float) -> Image.Image:
    alpha = np.array(mask, dtype=np.uint8)
    grad_rgb = _make_hard_split_gradient(mask.width, mask.height, c1, c2, angle=angle, alpha_mask=alpha)
    rgba = np.dstack([grad_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _add_rgb_noise(
    img_rgba: Image.Image,
    sigma: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Image.Image:
    arr = np.array(img_rgba, dtype=np.uint8)
    alpha = arr[:, :, 3] > 0
    if not np.any(alpha):
        return img_rgba
    noise = rng.normalvariate(0.0, sigma)
    # Per-image noise amplitude + per-pixel noise from seeded numpy RNG.
    np_noise = np_rng.normal(loc=noise, scale=sigma, size=arr[:, :, :3].shape)
    rgb = arr[:, :, :3].astype(np.float32)
    rgb[alpha] = np.clip(rgb[alpha] + np_noise[alpha], 0, 255)
    arr[:, :, :3] = rgb.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _sample_biased_scale(cfg: DatasetConfig, rng: random.Random) -> float:
    # Bias toward smaller values in the configured scale range.
    if rng.random() < cfg.tiny_scale_prob:
        return rng.uniform(*cfg.tiny_scale_range)
    s_min, s_max = cfg.scale_range
    # Higher bias -> stronger pull toward s_min.
    exponent = 1.0 + 4.0 * max(0.0, min(1.0, cfg.small_object_bias))
    t = rng.random() ** exponent
    return s_min + (s_max - s_min) * t


def apply_object_transforms(
    img_rgba: Image.Image,
    cfg: DatasetConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
    apply_appearance: bool = True,
    allow_flip: bool = True,
) -> Image.Image:
    # Draw both flip decisions even when flipping is disabled, so the random stream
    # (and therefore every other object in the dataset) does not depend on allow_flip.
    flip_h = rng.random() < cfg.flip_horizontal_prob
    flip_v = rng.random() < cfg.flip_vertical_prob
    if allow_flip and flip_h:
        img_rgba = ImageOps.mirror(img_rgba)
    if allow_flip and flip_v:
        img_rgba = ImageOps.flip(img_rgba)

    scale = _sample_biased_scale(cfg, rng)
    if rng.random() < cfg.anisotropic_stretch_prob:
        stretch_x = rng.uniform(*cfg.stretch_x_range)
        stretch_y = rng.uniform(*cfg.stretch_y_range)
    else:
        # Rare near-isotropic case.
        common = rng.uniform(0.85, 1.15)
        stretch_x = common
        stretch_y = common
    new_w = max(1, int(round(img_rgba.width * scale * stretch_x)))
    new_h = max(1, int(round(img_rgba.height * scale * stretch_y)))
    img_rgba = img_rgba.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    angle = rng.uniform(*cfg.rotation_range)
    img_rgba = img_rgba.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    max_w = max(1, int(round(cfg.image_width * cfg.max_object_width_ratio)))
    max_h = max(1, int(round(cfg.image_height * cfg.max_object_height_ratio)))
    if img_rgba.width > max_w or img_rgba.height > max_h:
        down = min(max_w / img_rgba.width, max_h / img_rgba.height)
        new_w = max(1, int(round(img_rgba.width * down)))
        new_h = max(1, int(round(img_rgba.height * down)))
        img_rgba = img_rgba.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    if apply_appearance:
        bright = rng.uniform(*cfg.object_brightness_range)
        contrast = rng.uniform(*cfg.object_contrast_range)
        rgb = img_rgba.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(bright)
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        img_rgba = Image.merge("RGBA", (*rgb.split(), img_rgba.split()[3]))

        if rng.random() < cfg.object_noise_prob:
            img_rgba = _add_rgb_noise(img_rgba, sigma=rng.uniform(4.0, 12.0), rng=rng, np_rng=np_rng)

        if rng.random() < cfg.object_sharpness_prob:
            sharpness = rng.uniform(*cfg.object_sharpness_range)
            rgb = img_rgba.convert("RGB")
            rgb = ImageEnhance.Sharpness(rgb).enhance(sharpness)
            img_rgba = Image.merge("RGBA", (*rgb.split(), img_rgba.split()[3]))

    return img_rgba


def crop_to_alpha(img_rgba: Image.Image) -> Optional[Image.Image]:
    alpha = np.array(img_rgba.split()[3])
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    return img_rgba.crop((x_min, y_min, x_max, y_max))


def get_tight_bbox_from_alpha(img_rgba: Image.Image, threshold: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """Inclusive (x_min, y_min, x_max, y_max) of pixels with alpha >= threshold."""
    alpha = np.array(img_rgba.split()[3])
    ys, xs = np.where(alpha >= max(1, threshold))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def prepare_background(
    background_path: Path,
    cfg: DatasetConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Image.Image:
    bg = Image.open(background_path).convert("RGB")
    if bg.size != (cfg.image_width, cfg.image_height):
        bg = bg.resize((cfg.image_width, cfg.image_height), resample=Image.Resampling.BICUBIC)

    if not cfg.background_augmentations_enabled:
        return bg

    bright = rng.uniform(*cfg.background_brightness_range)
    contrast = rng.uniform(*cfg.background_contrast_range)
    saturation = rng.uniform(*cfg.background_saturation_range)

    bg = ImageEnhance.Brightness(bg).enhance(bright)
    bg = ImageEnhance.Contrast(bg).enhance(contrast)
    bg = ImageEnhance.Color(bg).enhance(saturation)

    # Hue shift in HSV space.
    hsv = np.array(bg.convert("HSV"), dtype=np.uint8)
    shift = rng.randint(*cfg.background_hue_shift_range)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 256
    bg = Image.fromarray(hsv, mode="HSV").convert("RGB")

    # Gamma change.
    gamma = rng.uniform(*cfg.background_gamma_range)
    arr = np.array(bg, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0) ** (1.0 / gamma)
    arr = (arr * 255.0).astype(np.uint8)

    if rng.random() < cfg.background_noise_prob:
        noise_sigma = rng.uniform(2.0, 10.0)
        noise = np_rng.normal(0.0, noise_sigma, size=arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < cfg.background_tint_prob:
        tint = np.array(_random_color(rng), dtype=np.float32)
        alpha = rng.uniform(*cfg.background_tint_alpha_range)
        arr = np.clip(arr.astype(np.float32) * (1.0 - alpha) + tint[None, None, :] * alpha, 0, 255).astype(np.uint8)

    bg = Image.fromarray(arr, mode="RGB")
    if rng.random() < cfg.background_filter_prob:
        bg = _apply_background_distractor_filters(
            bg=bg,
            rng=rng,
            alpha_range=cfg.background_filter_alpha_range,
        )
    return bg


def _apply_background_distractor_filters(
    bg: Image.Image,
    rng: random.Random,
    alpha_range: Tuple[float, float],
) -> Image.Image:
    base = bg
    style = rng.choice(["unsharp", "edge", "emboss", "posterize", "solarize"])
    if style == "unsharp":
        filtered = base.filter(ImageFilter.UnsharpMask(radius=rng.uniform(0.9, 2.2), percent=rng.randint(140, 260), threshold=2))
    elif style == "edge":
        filtered = base.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif style == "emboss":
        filtered = base.filter(ImageFilter.EMBOSS)
    elif style == "posterize":
        filtered = ImageOps.posterize(base, bits=rng.randint(3, 5))
    else:
        filtered = ImageOps.solarize(base, threshold=rng.randint(96, 176))

    alpha = max(0.0, min(1.0, rng.uniform(*alpha_range)))
    if alpha <= 0.0:
        return base
    return Image.blend(base, filtered, alpha=alpha)
