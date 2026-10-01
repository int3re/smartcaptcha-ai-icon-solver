from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ALLOWED_EXT = {".png", ".jpg", ".jpeg"}


def validate_image_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXT:
        raise ValueError("Unsupported file type. Use .png, .jpg or .jpeg")


def get_image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        w, h = img.size
    return {"filename": path.name, "path": str(path), "width": w, "height": h}


def render_detections(image_path: Path, detections: list[dict[str, Any]]) -> Image.Image:
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for d in detections:
        b = d["bbox"]
        x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        label = f'{d["class_name"]} {d["confidence"]:.3f}'
        draw.rectangle((x1, y1, x2, y2), outline=(77, 163, 255), width=3)
        tw = len(label) * 7 + 8
        ty = max(0, y1 - 16)
        draw.rectangle((x1, ty, x1 + tw, ty + 14), fill=(19, 26, 46))
        draw.text((x1 + 4, ty + 1), label, font=font, fill=(220, 235, 255))
    return img


def render_detections_png(image_path: Path, detections: list[dict[str, Any]]) -> bytes:
    """Annotated image as PNG bytes, for showing in the UI without writing a file."""
    buf = io.BytesIO()
    render_detections(image_path, detections).save(buf, format="PNG")
    return buf.getvalue()


def draw_detections(image_path: Path, detections: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"annotated_{stamp}.png"
    render_detections(image_path, detections).save(out_path, format="PNG")
    return out_path
