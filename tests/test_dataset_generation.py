import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from config import DatasetConfig
from generate_dataset import prepare_output_dir
from utils.image_ops import (
    apply_object_transforms,
    build_class_map,
    choose_image_gradient,
    colorize_mask_with_gradient,
    crop_to_alpha,
    get_tight_bbox_from_alpha,
    load_object_assets,
)

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ROOT / "objects"


def load_assets(cfg):
    paths = sorted(OBJECTS.glob("*.png"))
    return {a.class_name: a for a in load_object_assets(OBJECTS, build_class_map(paths), cfg)}


class TightBBoxTests(unittest.TestCase):
    def test_tight_bbox_is_inside_crop_and_drops_fringe(self):
        cfg = DatasetConfig()
        assets = list(load_assets(cfg).values())
        rng, np_rng = random.Random(3), np.random.default_rng(3)
        shrunk = 0
        for _ in range(100):
            asset = rng.choice(assets)
            obj = colorize_mask_with_gradient(asset.mask, choose_image_gradient(cfg, rng))
            obj = crop_to_alpha(apply_object_transforms(obj, cfg, rng, np_rng))
            x0, y0, x1, y1 = get_tight_bbox_from_alpha(obj, threshold=cfg.bbox_alpha_threshold)
            self.assertTrue(0 <= x0 <= x1 < obj.width and 0 <= y0 <= y1 < obj.height)
            shrunk += (x1 - x0 + 1) < obj.width or (y1 - y0 + 1) < obj.height
        # Bicubic resampling leaves a faint fringe on nearly every transformed object.
        self.assertGreater(shrunk, 80)

    def test_threshold_one_matches_full_crop(self):
        cfg = DatasetConfig()
        asset = load_assets(cfg)["star"]
        obj = crop_to_alpha(colorize_mask_with_gradient(asset.mask, choose_image_gradient(cfg, random.Random(0))))
        self.assertEqual(get_tight_bbox_from_alpha(obj, threshold=1), (0, 0, obj.width - 1, obj.height - 1))


class FlipTests(unittest.TestCase):
    def _identity_cfg(self):
        return replace(
            DatasetConfig(),
            flip_horizontal_prob=1.0,
            flip_vertical_prob=0.0,
            rotation_range=(0.0, 0.0),
            scale_range=(1.0, 1.0),
            tiny_scale_prob=0.0,
            anisotropic_stretch_prob=1.0,
            stretch_x_range=(1.0, 1.0),
            stretch_y_range=(1.0, 1.0),
            max_object_width_ratio=10.0,
            max_object_height_ratio=10.0,
        )

    def test_no_flip_class_is_not_mirrored(self):
        cfg = self._identity_cfg()
        asset = load_assets(cfg)["a_word"]
        obj = colorize_mask_with_gradient(asset.mask, choose_image_gradient(cfg, random.Random(0)))
        alpha = lambda im: np.array(im.split()[3])
        kept = apply_object_transforms(obj, cfg, random.Random(1), np.random.default_rng(1), apply_appearance=False, allow_flip=False)
        flipped = apply_object_transforms(obj, cfg, random.Random(1), np.random.default_rng(1), apply_appearance=False, allow_flip=True)
        np.testing.assert_array_equal(alpha(kept), alpha(obj))
        self.assertFalse(np.array_equal(alpha(flipped), alpha(obj)))

    def test_allow_flip_does_not_change_random_stream(self):
        cfg = DatasetConfig()
        asset = load_assets(cfg)["a_word"]
        obj = colorize_mask_with_gradient(asset.mask, choose_image_gradient(cfg, random.Random(0)))
        rng_a, rng_b = random.Random(7), random.Random(7)
        apply_object_transforms(obj, cfg, rng_a, np.random.default_rng(7), allow_flip=True)
        apply_object_transforms(obj, cfg, rng_b, np.random.default_rng(7), allow_flip=False)
        self.assertEqual(rng_a.random(), rng_b.random())

    def test_default_no_flip_classes_exist(self):
        names = {p.stem for p in OBJECTS.glob("*.png")}
        self.assertTrue(set(DatasetConfig().no_flip_classes) <= names)


class OutputDirTests(unittest.TestCase):
    def test_refuses_protected_dir_and_its_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            with self.assertRaises(ValueError):
                prepare_output_dir(project, protected_dirs=[project])
            with self.assertRaises(ValueError):
                prepare_output_dir(Path(tmp), protected_dirs=[project])
            self.assertTrue(project.exists())

    def test_refuses_foreign_non_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "important.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_output_dir(out, protected_dirs=[])
            self.assertTrue((out / "important.txt").exists())

    def test_replaces_previous_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            (out / "train" / "images").mkdir(parents=True)
            (out / "data.yaml").write_text("nc: 1", encoding="utf-8")
            prepare_output_dir(out, protected_dirs=[])
            self.assertTrue(out.is_dir())
            self.assertEqual(list(out.iterdir()), [])


class GenerateCliTests(unittest.TestCase):
    def test_small_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds"
            cmd = [
                sys.executable, str(ROOT / "generate_dataset.py"),
                "--objects", str(OBJECTS), "--backgrounds", str(ROOT / "backgrounds"),
                "--output", str(out), "--count", "6", "--no-preview",
            ]
            subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)

            data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("path", data)
            self.assertEqual(data["nc"], 139)

            label_files = sorted(out.glob("*/labels_yolo/*.txt"))
            self.assertEqual(len(label_files), 6)
            for txt in label_files:
                ann = json.loads((txt.parents[1] / "labels_json" / f"{txt.stem}.json").read_text(encoding="utf-8"))
                lines = txt.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), len(ann["objects"]))
                self.assertGreaterEqual(len(lines), 6)
                for line, obj in zip(lines, ann["objects"]):
                    cls, xc, yc, w, h = line.split()
                    self.assertEqual(int(cls), obj["class_id"])
                    b = obj["bbox"]
                    self.assertAlmostEqual(float(w) * 320, b["x_max"] - b["x_min"] + 1, delta=0.01)
                    self.assertAlmostEqual(float(xc) * 320, b["x_min"] + (b["x_max"] - b["x_min"] + 1) / 2, delta=0.01)
                    self.assertTrue(all(0.0 < float(v) <= 1.0 for v in (xc, yc, w, h)))


if __name__ == "__main__":
    unittest.main()
