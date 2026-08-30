"""
scripts/make_sample_dataset.py
-------------------------------
Generates a small synthetic dataset (random grayscale-like images + random
multi-label targets) purely so the full pipeline — dataset loading,
augmentation, training loop, evaluation, inference, Grad-CAM, and the API —
can be smoke-tested end-to-end on a machine with no real medical dataset
downloaded yet.

This is NOT a substitute for real data. Replace dataset_sample/ with a real
NIH ChestX-ray14 or FracAtlas CSV + image directory before training a model
you intend to trust.

Usage:
    python scripts/make_sample_dataset.py --num-train 40 --num-val 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def make_split(n: int, image_dir: str, csv_path: str, seed: int):
    rng = np.random.default_rng(seed)
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for i in range(n):
        # Simple synthetic "radiograph-like" image: grayscale noise with a
        # brighter blob so Grad-CAM has something non-trivial to localize.
        img = rng.integers(20, 90, size=(512, 512), dtype=np.uint8)
        cx, cy = rng.integers(100, 412, size=2)
        radius = rng.integers(30, 80)
        yy, xx = np.ogrid[:512, :512]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        img[mask] = np.clip(img[mask].astype(int) + rng.integers(60, 120), 0, 255).astype(np.uint8)

        rgb = np.stack([img] * 3, axis=-1)
        filename = f"scan_{i:04d}.png"
        Image.fromarray(rgb).save(os.path.join(image_dir, filename))

        labels = rng.integers(0, 2, size=len(config.CLASS_NAMES))
        row = {"image_path": filename}
        row.update({name: int(val) for name, val in zip(config.CLASS_NAMES, labels)})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {n} samples -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=40)
    parser.add_argument("--num-val", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(config.IMAGE_ROOT, exist_ok=True)
    make_split(args.num_train, config.IMAGE_ROOT, config.TRAIN_CSV, seed=1)
    make_split(args.num_val, config.IMAGE_ROOT, config.VAL_CSV, seed=2)
    print("\nSample dataset ready. You can now run:\n  python train.py --epochs 2")


if __name__ == "__main__":
    main()
