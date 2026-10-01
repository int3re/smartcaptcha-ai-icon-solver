import argparse
import io
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageOps


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "ru,en-US;q=0.9,en;q=0.8",
            "Referer": "https://picsum.dev/",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def build_url(width: int, height: int, seed: str) -> str:
    return f"https://picsum.dev/{width}/{height}?seed={seed}"


def fetch_image(
    session: requests.Session,
    width: int,
    height: int,
    timeout: float,
) -> Image.Image:
    seed = uuid.uuid4().hex
    url = build_url(width, height, seed)

    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download random backgrounds and save them as uuid.png"
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument("--count", type=int, default=1000, help="Number of images")
    parser.add_argument("--width", type=int, default=320, help="Target width")
    parser.add_argument("--height", type=int, default=180, help="Target height")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout")
    parser.add_argument("--retries", type=int, default=5, help="Retries per image")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Sleep between successful downloads",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    saved = 0
    failures = 0
    max_failures = max(20, args.count // 10)

    while saved < args.count:
        success = False

        for attempt in range(1, args.retries + 1):
            try:
                img = fetch_image(
                    session=session,
                    width=args.width,
                    height=args.height,
                    timeout=args.timeout,
                )
                img = fit_image(img, args.width, args.height)

                filename = f"{uuid.uuid4()}.png"
                save_path = out_dir / filename
                img.save(save_path, format="PNG", optimize=True)

                saved += 1
                success = True
                print(f"[{saved}/{args.count}] saved {filename}")
                time.sleep(args.sleep)
                break

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                return 130

            except Exception as e:
                print(f"retry {attempt}/{args.retries} failed: {e}")
                time.sleep(min(1.0 * attempt, 3.0))

        if not success:
            failures += 1
            print(f"skip after {args.retries} retries; failures={failures}")

            if failures >= max_failures:
                print("too many failures, aborting")
                return 1

    print(f"done: saved={saved}, failures={failures}, output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())