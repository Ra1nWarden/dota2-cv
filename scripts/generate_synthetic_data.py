#!/usr/bin/env python3
"""Generate synthetic training data from raw Dota 2 icon PNGs.

Takes clean icon images and produces augmented variants suitable for
training an EfficientNet classifier. Outputs are organized into
train/val splits with ImageFolder-compatible directory structure.

Usage:
    python scripts/generate_synthetic_data.py \
        --icons data/raw_icons \
        --output data/synthetic \
        --samples-per-class 200 \
        --size 128 \
        --seed 42
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm


def composite_on_background(icon: Image.Image) -> Image.Image:
    """Composite an RGBA icon onto a random dark background."""
    bg_color = tuple(random.randint(10, 50) for _ in range(3))
    bg = Image.new("RGB", icon.size, bg_color)
    if icon.mode == "RGBA":
        bg.paste(icon, mask=icon.split()[3])
    else:
        bg.paste(icon)
    return bg


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square with dark fill, preserving aspect ratio."""
    w, h = image.size
    if w == h:
        return image
    size = max(w, h)
    bg_color = tuple(random.randint(10, 50) for _ in range(3))
    result = Image.new("RGB", (size, size), bg_color)
    offset = ((size - w) // 2, (size - h) // 2)
    result.paste(image, offset)
    return result


def apply_death_greyscale(image: Image.Image) -> Image.Image:
    """Desaturate and darken to simulate dead hero portrait.

    In-game, the death timer number appears below the portrait, outside
    the crop region, so we only desaturate + darken here.
    """
    grey = image.convert("L").convert("RGB")
    return ImageEnhance.Brightness(grey).enhance(random.uniform(0.5, 0.7))


def apply_cooldown_overlay(image: Image.Image) -> Image.Image:
    """Apply a dark radial cooldown overlay (items)."""
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Darken a pie-slice portion (random sweep angle from top)
    sweep = random.randint(60, 300)
    start_angle = -90
    draw.pieslice(
        [(-w // 4, -h // 4), (w + w // 4, h + h // 4)],
        start=start_angle,
        end=start_angle + sweep,
        fill=(0, 0, 0, random.randint(100, 160)),
    )

    image_rgba = image.convert("RGBA")
    composited = Image.alpha_composite(image_rgba, overlay)
    return composited.convert("RGB")


def add_gaussian_noise(image: Image.Image, sigma_range: tuple[float, float] = (0, 15)) -> Image.Image:
    """Add random Gaussian noise to image."""
    arr = np.array(image, dtype=np.float32)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_compress(image: Image.Image, quality_range: tuple[int, int] = (50, 95)) -> Image.Image:
    """Simulate JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_augmentation(image: Image.Image, category: str) -> Image.Image:
    """Apply the full augmentation pipeline to a single image."""
    img = image.copy()

    # Death greyscale (heroes only, 25% chance)
    is_hero = category == "heroes"
    if is_hero and random.random() < 0.25:
        img = apply_death_greyscale(img)

    # Cooldown overlay (items only, 30% chance)
    if category == "items" and random.random() < 0.30:
        img = apply_cooldown_overlay(img)

    # Random rotation ±5°
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    # Color jitter
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))

    # Gaussian noise
    img = add_gaussian_noise(img, sigma_range=(0, 15))

    # JPEG compression
    img = jpeg_compress(img, quality_range=(50, 95))

    return img


def generate_empty_samples(n: int, size: int) -> list[Image.Image]:
    """Generate empty slot samples (dark backgrounds with noise)."""
    samples = []
    for _ in range(n):
        bg_color = tuple(random.randint(5, 45) for _ in range(3))
        img = Image.new("RGB", (size, size), bg_color)
        # Add noise and slight texture variation
        img = add_gaussian_noise(img, sigma_range=(3, 20))
        img = jpeg_compress(img, quality_range=(60, 95))
        samples.append(img)
    return samples


def process_icon(icon_path: Path, size: int, category: str) -> Image.Image:
    """Load an icon, composite onto background, pad to square, resize."""
    icon = Image.open(icon_path)
    img = composite_on_background(icon)
    img = pad_to_square(img)
    img = img.resize((size, size), Image.LANCZOS)
    return img


def process_category(
    category: str,
    icons_dir: Path,
    output_dir: Path,
    samples_per_class: int,
    size: int,
    val_ratio: float = 0.15,
):
    """Generate synthetic data for one category (heroes or items)."""
    icon_files = sorted([
        f for f in icons_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    ])

    n_val = max(1, int(samples_per_class * val_ratio))
    n_train = samples_per_class - n_val

    for icon_path in tqdm(icon_files, desc=f"  {category}", unit="class"):
        class_name = icon_path.stem

        # Prepare base image
        base_img = process_icon(icon_path, size, category)

        for split, count in [("train", n_train), ("val", n_val)]:
            split_dir = output_dir / split / category / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                aug_img = apply_augmentation(base_img, category)
                aug_img.save(split_dir / f"{i:04d}.jpg", quality=95)

    # Generate empty class
    for split, count in [("train", n_train), ("val", n_val)]:
        empty_dir = output_dir / split / category / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

        empty_samples = generate_empty_samples(count, size)
        for i, img in enumerate(empty_samples):
            img.save(empty_dir / f"{i:04d}.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data from Dota 2 icons")
    parser.add_argument("--icons", required=True, help="Path to raw_icons directory")
    parser.add_argument("--output", required=True, help="Output directory for synthetic data")
    parser.add_argument("--samples-per-class", type=int, default=200, help="Augmented samples per class")
    parser.add_argument("--size", type=int, default=128, help="Output image size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    icons_dir = Path(args.icons)
    output_dir = Path(args.output)

    for category in ["heroes", "items"]:
        cat_dir = icons_dir / category
        if not cat_dir.is_dir():
            print(f"Warning: {cat_dir} not found, skipping")
            continue

        n_icons = len([f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
        print(f"\n{category}: {n_icons} icons x {args.samples_per_class} samples = {n_icons * args.samples_per_class} images")
        process_category(category, cat_dir, output_dir, args.samples_per_class, args.size)

    # Summary
    total = 0
    for split in ["train", "val"]:
        split_path = output_dir / split
        if split_path.exists():
            count = sum(1 for _ in split_path.rglob("*.jpg"))
            total += count
            print(f"\n{split}: {count} images")
    print(f"\nTotal: {total} images in {output_dir}")


if __name__ == "__main__":
    main()
