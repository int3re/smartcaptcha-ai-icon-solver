import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from train_yolo import parse_args, resolve_dataset_root, sync_ultralytics_labels


@contextlib.contextmanager
def chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def make_dataset(root: Path, splits=("train", "val")) -> Path:
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels_yolo").mkdir()
        for i in range(3):
            (root / split / "labels_yolo" / f"{i:06d}.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    data = {s: f"{s}/images" for s in splits}
    data.update(nc=1, names={0: "star"})
    if "train" not in data:
        data["train"] = data["val"] = data["test"]
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class ArgsTests(unittest.TestCase):
    def test_lr0_requires_explicit_optimizer(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--lr0", "0.001"])
        args = parse_args(["--lr0", "0.001", "--optimizer", "SGD", "--warmup-epochs", "0"])
        self.assertEqual((args.optimizer, args.lr0, args.warmup_epochs), ("SGD", 0.001, 0.0))

    def test_amp_unset_by_default(self):
        self.assertIsNone(parse_args([]).amp)
        self.assertFalse(parse_args(["--no-amp"]).amp)
        self.assertEqual(parse_args([]).fliplr, 0.0)


class DatasetRootTests(unittest.TestCase):
    def test_missing_path_resolves_next_to_yaml_from_any_cwd(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            data_yaml = make_dataset(Path(tmp) / "ds")
            with chdir(Path(other)):
                cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                self.assertEqual(resolve_dataset_root(data_yaml, cfg), data_yaml.parent.resolve())

    def test_ultralytics_resolves_same_root(self):
        from ultralytics.data.utils import check_det_dataset

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            data_yaml = make_dataset(Path(tmp) / "ds")
            with chdir(Path(other)):
                info = check_det_dataset(str(data_yaml), autodownload=False)
            self.assertEqual(Path(info["train"]).resolve(), (data_yaml.parent / "train" / "images").resolve())


class SyncTests(unittest.TestCase):
    def test_sync_links_labels_for_all_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            data_yaml = make_dataset(root, splits=("train", "val", "test"))
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                sync_ultralytics_labels(str(data_yaml), workers=2)
            for split in ("train", "val", "test"):
                self.assertEqual(len(list((root / split / "labels").glob("*.txt"))), 3)


if __name__ == "__main__":
    unittest.main()
