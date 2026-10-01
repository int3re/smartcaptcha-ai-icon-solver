import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from gui import chart_events as ce
from gui.image_utils import render_detections_png
from gui.model_utils import auto_detect_model, find_model_candidates, run_best_score
from gui.report_utils import build_report, filter_report, save_report_csv
from gui.state_store import load_state, update_state


def det(index, conf, name="star"):
    return {
        "index": index,
        "class_id": 0,
        "class_name": name,
        "confidence": conf,
        "bbox": {"x_min": 10, "y_min": 10, "x_max": 30, "y_max": 40},
        "width": 20,
        "height": 30,
        "area": 600,
        "center": {"x": 20, "y": 25},
    }


def make_run(root: Path, name: str, scores, mtime_offset: float = 0.0) -> Path:
    weights = root / "runs" / "train" / name / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"x")
    (weights / "last.pt").write_bytes(b"x")
    if scores is not None:
        lines = ["epoch,   metrics/mAP50(B),   metrics/mAP50-95(B)"]
        lines += [f"{i + 1},0.99,{s}" for i, s in enumerate(scores)]
        (weights.parent / "results.csv").write_text("\n".join(lines), encoding="utf-8")
    t = time.time() + mtime_offset
    os.utime(weights.parent, (t, t))
    return weights / "best.pt"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = build_report(
            image_info={"filename": "a.png", "path": "a.png", "width": 320, "height": 180},
            model_path=Path("m.pt"),
            detections=[det(1, 0.10), det(2, 0.50, "sun"), det(3, 0.90)],
            inference_ms=12.345,
            conf_thres=0.05,
            iou_thres=0.45,
            device="CPU",
        )

    def test_filter_report_matches_threshold(self):
        filtered = filter_report(self.report, 0.4)
        self.assertEqual([d["index"] for d in filtered["detections"]], [2, 3])
        self.assertEqual(filtered["summary"]["total_objects"], 2)
        self.assertEqual(filtered["summary"]["unique_classes"], ["star", "sun"])
        self.assertAlmostEqual(filtered["summary"]["avg_confidence"], 0.7)
        self.assertEqual(filtered["thresholds"]["confidence"], 0.4)
        self.assertEqual(filtered["thresholds"]["inference_confidence"], 0.05)
        # The stored report is left untouched for later threshold changes.
        self.assertEqual(len(self.report["detections"]), 3)
        self.assertEqual(self.report["thresholds"]["confidence"], 0.05)

    def test_csv_export_uses_filtered_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.csv"
            save_report_csv(filter_report(self.report, 0.8), out)
            rows = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2)  # header + one detection


class ModelCandidateTests(unittest.TestCase):
    def test_ranked_by_best_score_then_weights_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older_better = make_run(root, "a", [0.90, 0.939], mtime_offset=-100)
            newer_worse = make_run(root, "b", [0.93, 0.937])
            no_csv = make_run(root, "c", None, mtime_offset=50)
            weights = root / "weights"
            weights.mkdir()
            (weights / "yolov8n.pt").write_bytes(b"x")  # COCO base checkpoint, must be skipped
            (weights / "old_best.pt").write_bytes(b"x")

            candidates = find_model_candidates(root)
            self.assertEqual(candidates[0], older_better.resolve())
            self.assertEqual(candidates[2], newer_worse.resolve())
            self.assertEqual(candidates[4], no_csv.resolve())
            self.assertEqual(candidates[-1], (weights / "old_best.pt").resolve())
            self.assertNotIn((weights / "yolov8n.pt").resolve(), candidates)
            self.assertAlmostEqual(run_best_score(older_better.parents[1]), 0.939)

    def test_preferred_model_wins_when_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = make_run(root, "a", [0.9])
            manual = root / "manual.pt"
            manual.write_bytes(b"x")
            self.assertEqual(auto_detect_model(root, preferred=manual), manual.resolve())
            self.assertEqual(auto_detect_model(root, preferred=root / "missing.pt"), best.resolve())
            self.assertIsNone(auto_detect_model(root / "empty"))


class StateStoreTests(unittest.TestCase):
    def test_update_merges_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            update_state(path, theme="light")
            update_state(path, model_path="m.pt")
            self.assertEqual(load_state(path), {"theme": "light", "model_path": "m.pt"})

    def test_corrupt_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(load_state(path), {})


class ChartEventTests(unittest.TestCase):
    def test_index_from_attribute_and_payload(self):
        self.assertEqual(ce.event_index(SimpleNamespace(group_index=3), "group_index"), 3)
        payload = SimpleNamespace(data=json.dumps({"type": "tap", "detail": {"spot_index": 2}}))
        self.assertEqual(ce.event_index(payload, "group_index"), 2)
        self.assertEqual(ce.event_index(payload, "group_index", allow_payload=False), -1)

    def test_click_detection(self):
        self.assertTrue(ce.is_click_like_event(SimpleNamespace(type="tapUp")))
        self.assertFalse(ce.is_click_like_event(SimpleNamespace(type="pointerHover")))
        self.assertTrue(ce.is_chart_interaction_event(SimpleNamespace()))


class RenderTests(unittest.TestCase):
    def test_render_png_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "img.png"
            Image.new("RGB", (64, 48), "white").save(src)
            png = render_detections_png(src, [det(1, 0.9)])
            with Image.open(io.BytesIO(png)) as img:
                self.assertEqual(img.size, (64, 48))
                self.assertEqual(img.getpixel((10, 20)), (77, 163, 255))  # box outline


if __name__ == "__main__":
    unittest.main()
