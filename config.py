from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


Color = Tuple[int, int, int]
Palette = Tuple[Color, Color]


@dataclass
class DatasetConfig:
    image_width: int = 320
    image_height: int = 180
    dataset_size: int = 5000
    train_split: float = 0.8

    min_objects_per_image: int = 6
    max_objects_per_image: int = 8

    white_bg_threshold: int = 235
    min_object_size_px: int = 12
    edge_padding_px: int = 2

    rotation_range: Tuple[float, float] = (-90.0, 90.0)
    scale_range: Tuple[float, float] = (0.40, 0.90)
    tiny_scale_range: Tuple[float, float] = (0.3, 0.45)
    tiny_scale_prob: float = 0.12
    small_object_bias: float = 0.65
    stretch_x_range: Tuple[float, float] = (0.70, 1.45)
    stretch_y_range: Tuple[float, float] = (0.70, 1.45)
    anisotropic_stretch_prob: float = 0.72
    flip_horizontal_prob: float = 0.5
    flip_vertical_prob: float = 0.1
    object_noise_prob: float = 0.2
    object_brightness_range: Tuple[float, float] = (0.9, 1.15)
    object_contrast_range: Tuple[float, float] = (0.9, 1.15)
    object_sharpness_prob: float = 0.5
    object_sharpness_range: Tuple[float, float] = (1.1, 1.8)
    max_object_width_ratio: float = 0.34
    max_object_height_ratio: float = 0.58

    max_placement_attempts: int = 60
    max_scene_attempts: int = 25
    min_distance_between_boxes: int = 2

    balance_classes: bool = True

    background_augmentations_enabled: bool = True
    background_profile: str = "soft"
    background_brightness_range: Tuple[float, float] = (0.85, 1.2)
    background_contrast_range: Tuple[float, float] = (0.85, 1.2)
    background_saturation_range: Tuple[float, float] = (0.9, 1.2)
    background_hue_shift_range: Tuple[int, int] = (-8, 8)
    background_noise_prob: float = 0.2
    background_gamma_range: Tuple[float, float] = (0.9, 1.15)
    background_tint_prob: float = 0.25
    background_tint_alpha_range: Tuple[float, float] = (0.04, 0.16)
    background_filter_prob: float = 0.62
    background_filter_alpha_range: Tuple[float, float] = (0.3, 0.75)
    mono_black_bg_prob: float = 0.10
    mono_white_bg_prob: float = 0.10

    gradient_palettes: List[Palette] = field(
        default_factory=lambda: [
            ((255, 0, 212), (255, 42, 166)),    # fuchsia -> magenta
            ((255, 0, 200), (255, 79, 179)),    # fuchsia -> magenta (variant)
            ((255, 0, 200), (255, 90, 74)),     # magenta -> red/pink
            ((255, 42, 212), (255, 59, 107)),   # magenta -> red/pink (variant)
            ((138, 43, 255), (45, 107, 255)),   # purple -> blue
            ((160, 0, 255), (0, 168, 255)),     # purple -> blue (variant)
            ((57, 231, 255), (41, 121, 255)),   # cyan -> blue
            ((0, 216, 255), (47, 82, 255)),     # cyan -> blue (variant)
            ((83, 216, 255), (38, 93, 255)),    # light blue -> deep blue
            ((76, 207, 255), (42, 66, 255)),    # light blue -> deep blue (variant)
            ((215, 255, 0), (54, 255, 66)),     # lime -> green
            ((200, 255, 26), (0, 255, 87)),     # lime -> green (variant)
            ((243, 255, 0), (125, 255, 0)),     # yellow-green -> lime
            ((231, 255, 31), (65, 255, 0)),     # yellow-green -> lime (variant)
            ((255, 212, 0), (255, 138, 0)),     # yellow -> orange
            ((255, 223, 26), (255, 123, 0)),    # yellow -> orange (variant)
            ((255, 157, 0), (255, 75, 0)),      # orange -> red-orange
            ((255, 173, 31), (255, 92, 26)),    # orange -> red-orange (variant)
            ((255, 225, 0), (255, 90, 0)),      # yellow -> red-orange
            ((255, 216, 74), (255, 74, 31)),    # yellow -> red-orange (variant)
        ]
    )
    gradient_linear_prob: float = 0.35
    gradient_radial_prob: float = 0.20
    gradient_hard_split_prob: float = 0.35
    gradient_banded_prob: float = 0.10

    random_seed: int = 42
    save_preview: bool = True

    debug_mode: bool = False
    debug_dataset_size: int = 20
    visual_test_count: int = 200

    @classmethod
    def with_profile(cls, profile: str) -> "DatasetConfig":
        cfg = cls()
        if profile == "aggressive":
            cfg.background_brightness_range = (0.75, 1.35)
            cfg.background_contrast_range = (0.75, 1.35)
            cfg.background_saturation_range = (0.75, 1.35)
            cfg.background_hue_shift_range = (-16, 16)
            cfg.background_noise_prob = 0.35
            cfg.background_gamma_range = (0.8, 1.25)
            cfg.background_tint_prob = 0.4
            cfg.background_tint_alpha_range = (0.06, 0.22)
            cfg.background_filter_prob = 0.8
            cfg.background_filter_alpha_range = (0.4, 0.82)
            cfg.object_sharpness_prob = 0.6
            cfg.object_sharpness_range = (1.2, 2.0)
            cfg.background_profile = "aggressive"
        elif profile == "small_moderate":
            cfg.scale_range = (0.42, 0.92)
            cfg.tiny_scale_range = (0.30, 0.45)
            cfg.tiny_scale_prob = 0.12
            cfg.small_object_bias = 0.62
            cfg.stretch_x_range = (0.72, 1.35)
            cfg.stretch_y_range = (0.72, 1.35)
            cfg.anisotropic_stretch_prob = 0.65
            cfg.min_object_size_px = 12
            cfg.max_object_width_ratio = 0.30
            cfg.max_object_height_ratio = 0.50
            cfg.background_profile = "small_moderate"
        elif profile == "small_aggressive":
            cfg.scale_range = (0.32, 0.82)
            cfg.tiny_scale_range = (0.22, 0.38)
            cfg.tiny_scale_prob = 0.20
            cfg.small_object_bias = 0.80
            cfg.stretch_x_range = (0.58, 1.60)
            cfg.stretch_y_range = (0.58, 1.60)
            cfg.anisotropic_stretch_prob = 0.82
            cfg.min_object_size_px = 10
            cfg.max_object_width_ratio = 0.26
            cfg.max_object_height_ratio = 0.45
            cfg.background_profile = "small_aggressive"
        elif profile == "small_stretch_extreme":
            cfg.scale_range = (0.35, 0.88)
            cfg.tiny_scale_range = (0.24, 0.40)
            cfg.tiny_scale_prob = 0.16
            cfg.small_object_bias = 0.72
            cfg.stretch_x_range = (0.45, 1.85)
            cfg.stretch_y_range = (0.45, 1.85)
            cfg.anisotropic_stretch_prob = 0.90
            cfg.rotation_range = (-95.0, 95.0)
            cfg.min_object_size_px = 10
            cfg.max_object_width_ratio = 0.28
            cfg.max_object_height_ratio = 0.48
            cfg.background_profile = "small_stretch_extreme"
        return cfg
