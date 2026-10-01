from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8n.pt"
MODEL_PATH = WEIGHTS_DIR / MODEL_NAME

if not MODEL_PATH.exists():
    print("Downloading model...")
    model = YOLO(MODEL_NAME)
    downloaded_path = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
    if downloaded_path and downloaded_path.exists():
        MODEL_PATH.write_bytes(downloaded_path.read_bytes())
    else:
        model.save(str(MODEL_PATH))
    print("Model saved to:", MODEL_PATH)
else:
    print("Model already exists:", MODEL_PATH)
