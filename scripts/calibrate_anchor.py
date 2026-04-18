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
        --anchor-bbox 2150,1900,80,80

Output paths (relative to the workspace root, derived from --crop-config):
  configs/anchors/<anchor-name>_edges.png    # template PNG
  configs/anchor_offsets.json                # offsets metadata

The --anchor-bbox is x,y,w,h in the reference screenshot's pixel coordinates.
For a click-and-drag UI alternative, point your browser at /calibrate on the
labeling service.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the runtime helpers so the CLI and inference share one code path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference_service import (  # noqa: E402
    compute_canny_edges,
    compute_item_offsets,
    save_anchor_assets,
)


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

    img_bgr = cv2.imread(str(args.ref), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"error: failed to read image: {args.ref}", file=sys.stderr)
        return 2
    img_h, img_w = img_bgr.shape[:2]
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

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    edges = compute_canny_edges(img_rgb, (ax, ay, aw, ah), args.canny_low, args.canny_high)
    edge_density = float((edges > 0).sum()) / edges.size
    print(f"canny edges: {edge_density:.1%} edge density (low={args.canny_low}, high={args.canny_high})")
    if edge_density < 0.01:
        print(
            "warning: very few edges detected. Template may be too smooth to match reliably. "
            "Try lower Canny thresholds or a different anchor bbox.",
            file=sys.stderr,
        )

    item_offsets = compute_item_offsets(crop_config, ax, ay)
    if not item_offsets:
        print("error: no item_slot_* regions in crop_config", file=sys.stderr)
        return 2

    workspace = args.crop_config.resolve().parent.parent
    template_path, offsets_path = save_anchor_assets(
        workspace, edges, (ax, ay, aw, ah), item_offsets,
        anchor_name=args.anchor_name,
        match_threshold=args.match_threshold,
        canny_low=args.canny_low, canny_high=args.canny_high,
        reference_resolution=(ref_w, ref_h),
    )
    print(f"wrote template: {template_path} ({aw}x{ah})")
    print(f"wrote offsets: {offsets_path}")
    print()
    print("item offsets (relative to anchor top-left):")
    for name, off in item_offsets.items():
        print(f"  {name}: dx={off['dx']:+d} dy={off['dy']:+d} w={off['w']} h={off['h']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
