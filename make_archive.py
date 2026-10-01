"""Pack the project into a zip that is enough to reproduce generation, training and the GUI.

Python's zipfile marks non-ASCII names as UTF-8, so Cyrillic file names survive unpacking.
"""

import argparse
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CODE_GLOBS = ["*.py", "*.txt", "*.bat", "*.md", ".gitignore", "gui/*.py", "utils/*.py", "tests/*.py"]
# Everything an Ultralytics run produces except last.pt (only needed to resume training).
RUN_GLOBS = ["runs/train/*/*.yaml", "runs/train/*/*.csv", "runs/train/*/*.png", "runs/train/*/*.jpg", "runs/train/*/weights/best.pt"]


def collect(include_backgrounds: bool, extras: list[Path]) -> list[Path]:
    files: set[Path] = set()
    # weights/yolov8n.pt is the pretrained starting point, so training works offline too.
    for pattern in CODE_GLOBS + RUN_GLOBS + ["objects/*.png", "weights/yolov8n.pt"]:
        files.update(ROOT.glob(pattern))
    if include_backgrounds:
        files.update(ROOT.glob("backgrounds/*.png"))
    for extra in extras:
        extra = extra.resolve()
        if not extra.is_file():
            raise FileNotFoundError(f"Extra file not found: {extra}")
        files.add(extra)
    return sorted(p for p in files if p.is_file() and "__pycache__" not in p.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=f"YandexSmartAI_{date.today():%Y%m%d}.zip")
    parser.add_argument("--no-backgrounds", action="store_true", help="Skip backgrounds/ (~200 MB, already compressed PNGs).")
    parser.add_argument("--extra", nargs="*", default=[], help="Additional files to include, e.g. the presentation.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output archive if it exists.")
    args = parser.parse_args()

    out = Path(args.output).resolve()
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite or choose another --output.")

    files = collect(include_backgrounds=not args.no_backgrounds, extras=[Path(e) for e in args.extra])
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            if path == out:
                continue
            arcname = path.relative_to(ROOT).as_posix() if ROOT in path.parents else path.name
            # PNG/JPG/.pt are already compressed; storing them is much faster and just as small.
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".pt", ".pptx", ".docx"} else zipfile.ZIP_DEFLATED
            zf.write(path, arcname, compress_type=compress)

    print(f"Packed {len(files)} files into {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
