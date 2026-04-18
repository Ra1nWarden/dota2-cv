"""Calibrate anchor template + per-item offsets for anchor-relative cropping.

Given a reference screenshot, the existing crop_config.json, and a bounding
box around the anchor (the scepter/shard region between skills and items),
this script:

  1. Crops the anchor region from the reference screenshot.
  2. Runs Canny edge detection on the crop.
  3. Saves the edge image as the matching template.
  4. Computes (dx, dy) offsets from the anchor's top-left to each item slot's
     top-left, using the existing crop_config.json item coords.
  5. Writes configs/anchor_offsets.json with template path, match threshold,
     Canny thresholds, and per-item offsets.

The output files are then loaded at inference time by inference_service.py.

Usage:
    python scripts/calibrate_anchor.py \
        --ref data/reference_screenshot.png \
        --crop-config configs/crop_config.json \
        --anchor-bbox 2150,1900,80,80 \
        --template-out configs/anchors/scepter_edges.png \
        --offsets-out configs/anchor_offsets.json

The --anchor-bbox is x,y,w,h in the reference screenshot's pixel coordinates.
Use Preview (or any image viewer) to read pixel coords by drawing a marquee.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_bbox(s: str) -> tuple[int, int, int, int]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--anchor-bbox must be x,y,w,h (got {s!r})"
        )
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--anchor-bbox values must be ints (got {s!r})")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--anchor-bbox w and h must be positive")
    return x, y, w, h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, required=True, help="Reference screenshot path")
    p.add_argument("--crop-config", type=Path, required=True, help="crop_config.json path")
    p.add_argument(
        "--anchor-bbox",
        type=parse_bbox,
        required=True,
        help="Anchor bounding box: x,y,w,h in reference screenshot pixels",
    )
    p.add_argument(
        "--template-out",
        type=Path,
        default=Path("configs/anchors/scepter_edges.png"),
        help="Output path for the Canny edge template PNG",
    )
    p.add_argument(
        "--offsets-out",
        type=Path,
        default=Path("configs/anchor_offsets.json"),
        help="Output path for the offsets JSON",
    )
    p.add_argument("--canny-low", type=int, default=80, help="Canny low threshold")
    p.add_argument("--canny-high", type=int, default=160, help="Canny high threshold")
    p.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="Minimum match score below which inference falls back to fixed crops",
    )
    p.add_argument(
        "--anchor-name",
        default="scepter",
        help="Label written into the offsets JSON (informational only)",
    )
    args = p.parse_args()

    if not args.ref.exists():
        print(f"error: reference screenshot not found: {args.ref}", file=sys.stderr)
        return 2
    if not args.crop_config.exists():
        print(f"error: crop config not found: {args.crop_config}", file=sys.stderr)
        return 2

    img = cv2.imread(str(args.ref), cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: failed to read image: {args.ref}", file=sys.stderr)
        return 2
    img_h, img_w = img.shape[:2]
    print(f"reference: {args.ref} ({img_w}x{img_h})")

    with open(args.crop_config) as f:
        crop_config = json.load(f)

    ref_w, ref_h = crop_config["reference_resolution"]
    if (ref_w, ref_h) != (img_w, img_h):
        print(
            f"warning: reference screenshot is {img_w}x{img_h} but crop_config "
            f"reference is {ref_w}x{ref_h}. Anchor bbox is interpreted in screenshot coords."
        )

    ax, ay, aw, ah = args.anchor_bbox
    if ax < 0 or ay < 0 or ax + aw > img_w or ay + ah > img_h:
        print(
            f"error: anchor-bbox {args.anchor_bbox} is out of bounds for {img_w}x{img_h}",
            file=sys.stderr,
        )
        return 2
    print(f"anchor bbox (x,y,w,h): ({ax}, {ay}, {aw}, {ah})")

    # Crop and Canny
    crop = img[ay : ay + ah, ax : ax + aw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, args.canny_low, args.canny_high)
    edge_density = float((edges > 0).sum()) / edges.size
    print(f"canny edges: {edge_density:.1%} of pixels are edges (low={args.canny_low}, high={args.canny_high})")
    if edge_density < 0.01:
        print(
            "warning: very few edges detected. Template may be too smooth to match reliably. "
            "Try lower Canny thresholds or a different anchor bbox.",
            file=sys.stderr,
        )

    # Compute item offsets from existing crop_config (reference-resolution coords)
    # Note: anchor bbox is also in reference-resolution coords (assumed equal to screenshot dims).
    item_offsets: dict[str, dict[str, int]] = {}
    for name, region in crop_config["regions"].items():
        if not name.startswith("item_slot"):
            continue
        item_offsets[name] = {
            "dx": region["x"] - ax,
            "dy": region["y"] - ay,
            "w": region["w"],
            "h": region["h"],
        }
    if not item_offsets:
        print("error: no item_slot_* regions in crop_config", file=sys.stderr)
        return 2

    # Write template + offsets
    args.template_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.template_out), edges)
    print(f"wrote template: {args.template_out} ({aw}x{ah})")

    offsets = {
        "anchor": args.anchor_name,
        "template_path": str(args.template_out),
        "reference_resolution": [ref_w, ref_h],
        "anchor_bbox": {"x": ax, "y": ay, "w": aw, "h": ah},
        "match_threshold": args.match_threshold,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "item_offsets": item_offsets,
    }
    args.offsets_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.offsets_out, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"wrote offsets: {args.offsets_out}")
    print()
    print("item offsets (relative to anchor top-left):")
    for name, off in item_offsets.items():
        print(f"  {name}: dx={off['dx']:+d} dy={off['dy']:+d} w={off['w']} h={off['h']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
