from __future__ import annotations

from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def render_preview(base_image: Image.Image, objects: List[Dict]) -> Image.Image:
    preview = base_image.copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for obj in objects:
        b = obj["bbox"]
        x_min, y_min, x_max, y_max = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        label = obj["class"]
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 255, 255), width=2)
        txt_x, txt_y = x_min, max(0, y_min - 12)
        draw.rectangle((txt_x, txt_y, txt_x + len(label) * 6 + 6, txt_y + 11), fill=(0, 0, 0))
        draw.text((txt_x + 2, txt_y), label, fill=(255, 255, 0), font=font)
    return preview

