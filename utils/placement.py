from __future__ import annotations

import random
from typing import List, Optional, Tuple


BBox = Tuple[int, int, int, int]


def _inflate_bbox(b: BBox, margin: int) -> BBox:
    return (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def can_place_bbox(
    bbox: BBox,
    placed_bboxes: List[BBox],
    image_width: int,
    image_height: int,
    edge_padding: int,
    min_distance: int,
) -> bool:
    x_min, y_min, x_max, y_max = bbox
    if x_min < edge_padding or y_min < edge_padding:
        return False
    if x_max > image_width - 1 - edge_padding or y_max > image_height - 1 - edge_padding:
        return False

    current = _inflate_bbox(bbox, min_distance)
    for other in placed_bboxes:
        if bboxes_intersect(current, _inflate_bbox(other, min_distance)):
            return False
    return True


def find_position_for_object(
    obj_w: int,
    obj_h: int,
    placed_bboxes: List[BBox],
    image_width: int,
    image_height: int,
    max_attempts: int,
    edge_padding: int,
    min_distance: int,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    if obj_w <= 0 or obj_h <= 0:
        return None

    max_x = image_width - edge_padding - obj_w
    max_y = image_height - edge_padding - obj_h
    min_x = edge_padding
    min_y = edge_padding
    if max_x < min_x or max_y < min_y:
        return None

    for _ in range(max_attempts):
        x = rng.randint(min_x, max_x)
        y = rng.randint(min_y, max_y)
        bbox = (x, y, x + obj_w - 1, y + obj_h - 1)
        if can_place_bbox(
            bbox=bbox,
            placed_bboxes=placed_bboxes,
            image_width=image_width,
            image_height=image_height,
            edge_padding=edge_padding,
            min_distance=min_distance,
        ):
            return x, y
    return None

