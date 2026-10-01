import argparse
import shutil
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

WEIGHTS_DIR = Path("weights")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained YOLO weights into weights/.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics asset name, e.g. yolov8n.pt.")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    model_path = WEIGHTS_DIR / args.model
    if model_path.exists():
        print("Model already exists:", model_path)
        return

    print("Downloading model...")
    # Downloads straight to weights/<name>, so no extra copy is left in the project root.
    found = Path(attempt_download_asset(model_path))
    if not model_path.exists() and found.is_file():
        # Already present in Ultralytics' global weights_dir: copy it into the project.
        shutil.copy2(found, model_path)
    if not model_path.exists():
        raise RuntimeError(f"Download failed: {model_path}")
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
