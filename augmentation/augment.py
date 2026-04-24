#!/usr/bin/env python3
"""
SUITS-VLM-Bench — Augmentation Script
--------------------------------------
Reads source images from data/{egress,navigation,ltv_repair}/
Outputs augmented images + manifest.json to output/augmented/

Usage:
    python augment.py
    python augment.py --copies 15 --size 1024
    python augment.py --copies 10 --size 768 --no-cutout --no-noise
    python augment.py --source data/ --output output/augmented/

Dependencies:
    pip install Pillow numpy
"""

import os
import json
import random
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SUITS-VLM-Bench image augmentation pipeline")
    p.add_argument("--source",    default="data",             help="Root folder containing egress/, navigation/, ltv_repair/ subfolders")
    p.add_argument("--output",    default="output/augmented", help="Output folder for augmented images and manifest")
    p.add_argument("--copies",    type=int,   default=10,     help="Augmented copies per source image (default: 10)")
    p.add_argument("--size",      type=int,   default=1024,   help="Output image size in pixels, 0 = keep original (default: 1024)")
    p.add_argument("--seed",      type=int,   default=None,   help="Random seed for reproducibility")
    p.add_argument("--quality",   type=int,   default=92,     help="JPEG output quality 1-95 (default: 92)")

    # Geometric
    p.add_argument("--no-flip-h",    action="store_true", help="Disable horizontal flip")
    p.add_argument("--no-flip-v",    action="store_true", help="Disable vertical flip")
    p.add_argument("--no-rotate",    action="store_true", help="Disable random rotation")
    p.add_argument("--rot-max",  type=float, default=25.0,   help="Max rotation angle in degrees (default: 25)")
    p.add_argument("--no-crop",      action="store_true", help="Disable random crop")
    p.add_argument("--no-translate", action="store_true", help="Disable random translation")

    # Photometric
    p.add_argument("--no-jitter",    action="store_true", help="Disable color jitter")
    p.add_argument("--jitter-str",   type=float, default=0.30, help="Color jitter strength 0-1 (default: 0.30)")
    p.add_argument("--no-blur",      action="store_true", help="Disable Gaussian blur")
    p.add_argument("--no-noise",     action="store_true", help="Disable random noise")
    p.add_argument("--no-grayscale", action="store_true", help="Disable stochastic grayscale")

    # Masking
    p.add_argument("--no-cutout",    action="store_true", help="Disable cutout patch")
    p.add_argument("--cutout-size",  type=float, default=0.22, help="Cutout patch size as fraction of image (default: 0.22)")
    p.add_argument("--no-erasing",   action="store_true", help="Disable random erasing")
    p.add_argument("--no-gridmask",  action="store_true", help="Disable GridMask")

    return p.parse_args()


# ── AUGMENTATION ─────────────────────────────────────────────────────────────

def rnd(a, b):
    return a + random.random() * (b - a)


def augment_image(img: Image.Image, args) -> Image.Image:
    """Apply a random combination of enabled augmentations to a PIL image."""
    w, h = img.size

    # ── Geometric ──────────────────────────────────────────────────────────

    if not args.no_flip_h and random.random() > 0.5:
        img = ImageOps.mirror(img)

    if not args.no_flip_v and random.random() > 0.7:
        img = ImageOps.flip(img)

    if not args.no_rotate:
        angle = rnd(-args.rot_max, args.rot_max)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    if not args.no_translate and random.random() > 0.5:
        tx = int(rnd(-w * 0.08, w * 0.08))
        ty = int(rnd(-h * 0.08, h * 0.08))
        img = img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), resample=Image.BICUBIC)

    if not args.no_crop and random.random() > 0.4:
        scale = rnd(0.78, 0.95)
        cw, ch = int(w * scale), int(h * scale)
        x0 = int(rnd(0, w - cw))
        y0 = int(rnd(0, h - ch))
        img = img.crop((x0, y0, x0 + cw, y0 + ch))

    # ── Photometric ────────────────────────────────────────────────────────

    if not args.no_jitter:
        s = args.jitter_str
        img = ImageEnhance.Brightness(img).enhance(rnd(1 - s, 1 + s))
        img = ImageEnhance.Contrast(img).enhance(rnd(1 - s * 0.6, 1 + s * 0.6))
        img = ImageEnhance.Color(img).enhance(rnd(1 - s, 1 + s))
        img = ImageEnhance.Sharpness(img).enhance(rnd(0.5, 2.0))

    if not args.no_blur and random.random() > 0.5:
        radius = rnd(0.5, 2.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if not args.no_grayscale and random.random() > 0.85:
        img = ImageOps.grayscale(img).convert("RGB")

    if not args.no_noise and random.random() > 0.6:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, rnd(5, 18), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # ── Resize to target ───────────────────────────────────────────────────

    if args.size > 0:
        img = img.resize((args.size, args.size), Image.LANCZOS)
    out_w, out_h = img.size

    # ── Masking ────────────────────────────────────────────────────────────

    arr = np.array(img)

    if not args.no_cutout and random.random() > 0.4:
        cs = args.cutout_size
        cw = int(out_w * rnd(cs * 0.7, cs * 1.3))
        ch = int(out_h * rnd(cs * 0.7, cs * 1.3))
        x0 = int(rnd(0, out_w - cw))
        y0 = int(rnd(0, out_h - ch))
        arr[y0:y0 + ch, x0:x0 + cw] = 0

    if not args.no_erasing and random.random() > 0.6:
        ew = int(rnd(out_w * 0.10, out_w * 0.25))
        eh = int(rnd(out_h * 0.10, out_h * 0.25))
        ex = int(rnd(0, out_w - ew))
        ey = int(rnd(0, out_h - eh))
        arr[ey:ey + eh, ex:ex + ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

    if not args.no_gridmask and random.random() > 0.7:
        gs = random.randint(out_w // 10, out_w // 6)
        ratio = rnd(0.3, 0.5)
        for gy in range(0, out_h, gs):
            for gx in range(0, out_w, gs):
                if random.random() > 0.5:
                    gw2 = int(gs * ratio)
                    gh2 = int(gs * ratio)
                    arr[gy:min(gy + gh2, out_h), gx:min(gx + gw2, out_w)] = 0

    return Image.fromarray(arr)


# ── MAIN ─────────────────────────────────────────────────────────────────────

PHASE_DIRS = {
    "egress":     "EGRESS",
    "navigation": "NAVIGATION",
    "ltv_repair": "LTV_REPAIR",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source_root = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect source images
    source_images = []
    for dir_name, phase_label in PHASE_DIRS.items():
        phase_dir = source_root / dir_name
        if not phase_dir.exists():
            print(f"  [skip] {phase_dir} not found")
            continue
        found = [f for f in phase_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        print(f"  [found] {len(found)} images in {phase_dir} → label: {phase_label}")
        for f in found:
            source_images.append((f, phase_label))

    if not source_images:
        print("\nNo source images found. Add images to:")
        for d in PHASE_DIRS:
            print(f"  {source_root / d}/")
        return

    total = len(source_images) * args.copies
    print(f"\nAugmenting {len(source_images)} source images × {args.copies} copies = {total} total")
    print(f"Output size: {'original' if args.size == 0 else f'{args.size}x{args.size}px'}")
    print(f"Output dir:  {output_root}\n")

    manifest = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "config": {
            "copies": args.copies,
            "size": args.size if args.size > 0 else "original",
            "quality": args.quality,
            "seed": args.seed,
        },
        "total": total,
        "images": []
    }

    done = 0
    errors = 0

    for src_path, phase in source_images:
        stem = src_path.stem
        try:
            orig = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [error] Could not open {src_path.name}: {e}")
            errors += 1
            continue

        # Create phase subfolder in output
        phase_out = output_root / phase.lower()
        phase_out.mkdir(parents=True, exist_ok=True)

        for i in range(args.copies):
            out_name = f"{phase}_{stem}_aug_{str(i + 1).zfill(3)}.jpg"
            out_path = phase_out / out_name

            try:
                aug = augment_image(orig, args)
                aug.save(out_path, "JPEG", quality=args.quality)

                manifest["images"].append({
                    "filename": out_name,
                    "path": str(out_path.relative_to(output_root)),
                    "phase": phase,
                    "source_image": src_path.name,
                    "resolution": aug.size[0],
                    "copy_index": i + 1,
                })

                done += 1
                print(f"  [{done}/{total}] {out_name}", end="\r", flush=True)

            except Exception as e:
                print(f"\n  [error] {out_name}: {e}")
                errors += 1

    print(f"\n\nDone. {done} images generated, {errors} errors.")

    # Save manifest
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")

    # Summary
    print("\n── Summary ───────────────────────────────────────")
    for phase in PHASE_DIRS.values():
        count = sum(1 for img in manifest["images"] if img["phase"] == phase)
        print(f"  {phase:<15} {count} images")
    print(f"  {'TOTAL':<15} {done} images")
    print("──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()