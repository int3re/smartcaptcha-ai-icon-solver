"""Illustrations for the README, built with the project's own generator functions.

Run from the project root:  python docs/make_figures.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DatasetConfig  # noqa: E402
from generate_dataset import generate_one_image  # noqa: E402
from utils.image_ops import (  # noqa: E402
    GradientSpec,
    apply_object_transforms,
    build_class_map,
    colorize_mask_with_duotone,
    colorize_mask_with_gradient,
    crop_to_alpha,
    get_tight_bbox_from_alpha,
    load_background_paths,
    load_object_assets,
    prepare_background,
)

OUT = ROOT / "docs" / "images"
BG = (13, 17, 23)  # GitHub dark background
FG = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (77, 163, 255)
BOX = (255, 214, 0)


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def on_white(img: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img.convert("RGBA"))
    return canvas.convert("RGB")


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    # Strong upscaling of tiny objects stays pixel-exact instead of turning into blur.
    resample = Image.Resampling.NEAREST if scale > 2 else Image.Resampling.LANCZOS
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), resample)


def panel_row(tiles: list[tuple[Image.Image, str]], tile_w: int, tile_h: int, title: str = "", arrows: bool = False) -> Image.Image:
    pad, cap_h, title_h, gap = 24, 52, (56 if title else 0), (44 if arrows else 20)
    width = pad * 2 + len(tiles) * tile_w + (len(tiles) - 1) * gap
    height = pad * 2 + title_h + tile_h + cap_h
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, pad), title, font=font(26), fill=FG)
    y0 = pad + title_h
    for i, (img, caption) in enumerate(tiles):
        x0 = pad + i * (tile_w + gap)
        draw.rounded_rectangle((x0, y0, x0 + tile_w, y0 + tile_h), radius=10, fill=(22, 27, 34), outline=(48, 54, 61))
        shown = fit(img, tile_w - 16, tile_h - 16)
        sheet.paste(shown, (x0 + (tile_w - shown.width) // 2, y0 + (tile_h - shown.height) // 2))
        for j, line in enumerate(caption.split("\n")):
            f = font(18 if j == 0 else 15)
            tw = draw.textlength(line, font=f)
            draw.text((x0 + (tile_w - tw) / 2, y0 + tile_h + 8 + j * 22), line, font=f, fill=FG if j == 0 else MUTED)
        if arrows and i < len(tiles) - 1:
            ax, ay = x0 + tile_w + gap / 2, y0 + tile_h / 2
            draw.polygon([(ax - 9, ay - 11), (ax + 11, ay), (ax - 9, ay + 11)], fill=ACCENT)
    return sheet


def draw_boxes(img: Image.Image, objects: list[dict], scale: int, labels: bool = True) -> Image.Image:
    big = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(big)
    f = font(13)
    for obj in objects:
        b = obj["bbox"]
        x0, y0 = b["x_min"] * scale, b["y_min"] * scale
        x1, y1 = (b["x_max"] + 1) * scale - 1, (b["y_max"] + 1) * scale - 1
        draw.rectangle((x0, y0, x1, y1), outline=BOX, width=2)
        if labels:
            tw = draw.textlength(obj["class"], font=f)
            ty = max(0, y0 - 18)
            draw.rectangle((x0, ty, x0 + tw + 8, ty + 17), fill=(0, 0, 0))
            draw.text((x0 + 4, ty + 1), obj["class"], font=f, fill=BOX)
    return big


def load_assets(cfg: DatasetConfig):
    objects_dir = ROOT / "objects"
    class_map = build_class_map(sorted(objects_dir.glob("*.png")))
    return {a.class_name: a for a in load_object_assets(objects_dir, class_map, cfg)}


def fig_generation_steps(cfg, assets, backgrounds) -> None:
    rng, np_rng = random.Random(11), np.random.default_rng(11)
    asset = assets["rocket"]
    original = on_white(Image.open(asset.path))
    mask = asset.mask.convert("RGB")
    gradient = GradientSpec(c1=(255, 0, 212), c2=(255, 42, 166), kind="hard_split", angle=0.6, center_x=0.5, center_y=0.5)
    colored = colorize_mask_with_gradient(asset.mask, gradient)
    transformed = crop_to_alpha(apply_object_transforms(colored, cfg, rng, np_rng))
    bg = prepare_background(backgrounds[5], cfg, rng, np_rng).convert("RGBA")
    x, y = 120, 40
    bg.alpha_composite(transformed, dest=(x, y))
    tx0, ty0, tx1, ty1 = get_tight_bbox_from_alpha(transformed, cfg.bbox_alpha_threshold)
    scene = draw_boxes(
        bg.convert("RGB"),
        [{"class": asset.class_name, "bbox": {"x_min": x + tx0, "y_min": y + ty0, "x_max": x + tx1, "y_max": y + ty1}}],
        scale=2,
    )
    tiles = [
        (original, "1. Исходная иконка\nчёрная на белом, 80×80"),
        (mask, "2. Маска\nпиксели темнее 235"),
        (on_white(colored), "3. Раскраска\nградиент из палитры"),
        (on_white(transformed), "4. Трансформации\nмасштаб, растяжение, поворот"),
        (scene, "5. Вставка на фон\n+ рамка YOLO"),
    ]
    panel_row(tiles, 280, 220, "Как из одной иконки получается размеченный объект", arrows=True).save(OUT / "generation_steps.png")


def fig_colorization(assets) -> None:
    mask = assets["shop_cart"].mask
    c1, c2 = (160, 0, 255), (0, 168, 255)
    kinds = [
        (GradientSpec(c1, c2, "linear", 0.0, 0.5, 0.5), "Линейный", "35 %"),
        (GradientSpec(c1, c2, "radial", 0.0, 0.55, 0.55), "Радиальный", "20 %"),
        (GradientSpec(c1, c2, "hard_split", 0.8, 0.5, 0.5), "Две половины", "35 %"),
        (GradientSpec(c1, c2, "banded", 0.8, 0.5, 0.5), "Полосы", "10 %"),
    ]
    tiles = [(on_white(colorize_mask_with_gradient(mask, spec)), f"{name}\n{share} обычных сцен") for spec, name, share in kinds]
    black = Image.new("RGBA", mask.size, (0, 0, 0, 255))
    black.alpha_composite(colorize_mask_with_duotone(mask, (220, 220, 220), (255, 255, 255), 0.8))
    tiles.append((black.convert("RGB"), "Белое на чёрном\n10 % сцен"))
    tiles.append((on_white(colorize_mask_with_duotone(mask, (0, 0, 0), (46, 46, 46), 0.8)), "Чёрное на белом\n10 % сцен"))
    panel_row(tiles, 200, 170, "Способы раскраски (20 двухцветных палитр)").save(OUT / "colorization.png")


def fig_backgrounds(cfg, backgrounds) -> None:
    src = Image.open(backgrounds[42]).convert("RGB")
    rng, np_rng = random.Random(5), np.random.default_rng(5)
    soft = replace(cfg, background_filter_prob=0.0)
    tiles = [(src, "Исходный фон\n320×180")]
    tiles += [(prepare_background(backgrounds[42], soft, rng, np_rng), "Яркость, контраст,\nоттенок, гамма, шум")]
    styles = [
        ("Unsharp", src.filter(ImageFilter.UnsharpMask(radius=1.8, percent=220, threshold=2))),
        ("Edge", src.filter(ImageFilter.EDGE_ENHANCE_MORE)),
        ("Emboss", src.filter(ImageFilter.EMBOSS)),
        ("Posterize", ImageOps.posterize(src, 3)),
        ("Solarize", ImageOps.solarize(src, 128)),
    ]
    tiles += [(Image.blend(src, filtered, 0.65), f"{name}\nфильтр-отвлекатель") for name, filtered in styles]
    panel_row(tiles, 250, 150, "Аугментация фонов: 62 % фонов получают фильтр, похожий на штрихи иконок").save(OUT / "backgrounds.jpg", quality=90)


def fig_samples(cfg, assets_list, backgrounds) -> None:
    rng, np_rng = random.Random(2026), np.random.default_rng(2026)
    usage = {a.class_name: 0 for a in assets_list}
    scenes = []
    for i in range(9):
        img, objects = generate_one_image(i, cfg, assets_list, backgrounds, usage, rng, np_rng)
        scenes.append(draw_boxes(img, objects, scale=2))
    w, h, gap, pad = scenes[0].width, scenes[0].height, 12, 24
    sheet = Image.new("RGB", (pad * 2 + 3 * w + 2 * gap, pad * 2 + 56 + 3 * h + 2 * gap), BG)
    ImageDraw.Draw(sheet).text((pad, pad), "Примеры сгенерированных изображений с разметкой (6–8 разных классов на кадр)", font=font(26), fill=FG)
    for i, scene in enumerate(scenes):
        sheet.paste(scene, (pad + (i % 3) * (w + gap), pad + 56 + (i // 3) * (h + gap)))
    sheet.save(OUT / "dataset_samples.jpg", quality=88)


def fig_bbox_fix(cfg, assets) -> None:
    rng, np_rng = random.Random(4), np.random.default_rng(4)
    gradient = GradientSpec((0, 216, 255), (47, 82, 255), "linear", 1.1, 0.5, 0.5)
    obj = crop_to_alpha(apply_object_transforms(colorize_mask_with_gradient(assets["key"].mask, gradient), cfg, rng, np_rng))
    scale = 10
    base = on_white(obj).resize((obj.width * scale, obj.height * scale), Image.Resampling.NEAREST)

    def boxed(threshold: int, color) -> Image.Image:
        img = base.copy()
        x0, y0, x1, y1 = get_tight_bbox_from_alpha(obj, threshold)
        ImageDraw.Draw(img).rectangle((x0 * scale, y0 * scale, (x1 + 1) * scale - 1, (y1 + 1) * scale - 1), outline=color, width=3)
        return img

    old = get_tight_bbox_from_alpha(obj, 1)
    new = get_tight_bbox_from_alpha(obj, cfg.bbox_alpha_threshold)
    tiles = [
        (boxed(1, (220, 38, 38)), f"Было: alpha > 0\n{old[2] - old[0] + 1}×{old[3] - old[1] + 1} px, захватывает кайму"),
        (boxed(cfg.bbox_alpha_threshold, (22, 163, 74)), f"Стало: alpha ≥ {cfg.bbox_alpha_threshold}\n{new[2] - new[0] + 1}×{new[3] - new[1] + 1} px, по видимому объекту"),
    ]
    panel_row(tiles, 420, 360, "Точность рамок: кайма от интерполяции").save(OUT / "bbox_fix.png")


def read_results(run: str) -> dict[str, list[float]]:
    import csv

    with (ROOT / "runs" / "train" / run / "results.csv").open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return {k: [r[k] for r in rows] for k in rows[0]}


def fig_training() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, grid, ink, muted = "#0d1117", "#30363d", "#e6edf3", "#8b949e"
    base_color, ft_color = "#3987e5", "#d95926"  # categorical slots 1-2, validated on this surface
    base = read_results("yolov8n_50k_maxspeed2")
    ft = read_results("yolov8n_50k_finetune_20260324")
    n_base = len(base["epoch"])
    x_base = base["epoch"]
    x_ft = [n_base + e for e in ft["epoch"]]  # fine-tuning continues after the base run

    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1]})
    fig.patch.set_facecolor(surface)
    panels = [
        ("metrics/mAP50-95(B)", "mAP50-95 (val)", (0.84, 0.95)),
        ("train/cls_loss", "Потери классификации (train)", None),
        ("lr/pg0", "Скорость обучения", None),
    ]
    for ax, (key, label, ylim) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.plot(x_base, base[key], color=base_color, lw=2, label="Основное обучение (40 эпох)")
        ax.plot(x_ft, ft[key], color=ft_color, lw=2, label="Дообучение на той же выборке (20 эпох)")
        ax.set_ylabel(label, color=muted)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        for x in (31, n_base + 11):
            ax.axvline(x, color=grid, lw=1, ls="--")

    top = axes[0]
    top.set_title("Обучение YOLOv8n на 50 000 синтетических изображений", color=ink, loc="left", fontsize=15, pad=34)
    top.legend(loc="lower right", frameon=False, labelcolor=ink, bbox_to_anchor=(1.0, 1.0), ncol=2)
    best = max(base["metrics/mAP50-95(B)"])
    top.annotate(f"{best:.3f}", (x_base[-1], base["metrics/mAP50-95(B)"][-1]), xytext=(-6, 8), textcoords="offset points", color=ink, ha="right")
    dip = min(ft["metrics/mAP50-95(B)"])
    dip_x = x_ft[ft["metrics/mAP50-95(B)"].index(dip)]
    top.annotate(
        f"LR снова прогрет до максимума:\nпровал до {dip:.3f}",
        (dip_x, dip),
        xytext=(n_base + 12, 0.895),
        color=muted,
        arrowprops=dict(arrowstyle="->", color=muted, lw=1),
    )
    top.annotate(f"{ft['metrics/mAP50-95(B)'][-1]:.3f}", (x_ft[-1], ft["metrics/mAP50-95(B)"][-1]), xytext=(-6, 8), textcoords="offset points", color=ink, ha="right")
    axes[1].annotate("mosaic выключен", (31, max(base["train/cls_loss"])), xytext=(6, -2), textcoords="offset points", color=muted, va="top")
    axes[1].annotate("mosaic выключен", (n_base + 11, max(base["train/cls_loss"])), xytext=(6, -2), textcoords="offset points", color=muted, va="top")
    axes[2].set_xlabel("Эпоха", color=muted)
    axes[2].set_xticks([1, 10, 20, 31, 40, 50, 60])
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=110, facecolor=surface)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_training()
    cfg = DatasetConfig()
    assets = load_assets(cfg)
    backgrounds = load_background_paths(ROOT / "backgrounds")
    fig_generation_steps(cfg, assets, backgrounds)
    fig_colorization(assets)
    fig_backgrounds(cfg, backgrounds)
    fig_samples(cfg, list(assets.values()), backgrounds)
    fig_bbox_fix(cfg, assets)
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size / 1024:8.0f} KB  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
