"""Quick smoke-test for /predict: prints hero_name + items for each test screenshot.

Usage:
    python scripts/test_ocr.py --host http://<truenas-ip>:18080
    python scripts/test_ocr.py --host http://<truenas-ip>:18080 --dir data/test_screenshots
"""

import argparse
import json
from pathlib import Path

import requests


def predict(host: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        r = requests.post(f"{host}/predict", files={"file": f}, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:18080")
    parser.add_argument("--dir", default="data/test_screenshots")
    args = parser.parse_args()

    screenshots = sorted(Path(args.dir).glob("*.png"))
    if not screenshots:
        print(f"No .png files found in {args.dir}")
        return

    for path in screenshots:
        print(f"\n{'─'*60}")
        print(f"  {path.name}")
        print(f"{'─'*60}")
        try:
            result = predict(args.host, path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        hero_name = result.get("hero_name")
        anchor = result.get("anchor", {})
        print(f"  hero_name : {hero_name or '(not identified)'}")
        print(f"  anchor    : used={anchor.get('used')} score={anchor.get('score')}")

        items = result.get("items", {})
        if items:
            print("  items:")
            for slot, info in sorted(items.items()):
                print(f"    {slot:12s}  {info['class']:30s}  conf={info['confidence']:.2f}")
        else:
            print("  items: (none)")


if __name__ == "__main__":
    main()
