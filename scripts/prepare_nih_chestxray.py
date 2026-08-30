"""
scripts/prepare_nih_chestxray.py
----------------------------------
Converts a (possibly partial) local copy of the NIH ChestX-ray14 dataset
into the CSV format expected by data/dataset.py.

NIH ChestX-ray14 is genuinely multi-label across 14 findings, and the full
image set is ~42GB — impractical to fully download/train on a CPU-only
machine. This script is built to work against however many images you've
actually downloaded: it filters the metadata CSV down to only the images
present on disk, so you can start with a small slice and grow later.

--------------------------------------------------------------------------
STEP 1 — Download the metadata CSV (small, do this first):
    From the NIH Box release: https://nihcc.app.box.com/v/ChestXray-NIHCC
    Grab "Data_Entry_2017_v2020.csv" (or "Data_Entry_2017.csv").

STEP 2 — Download image archives (each ~2GB, ~10,000 images):
    Same Box link has "images_001.zip" through "images_012.zip".
    You do NOT need all 12 — start with 1-2 for a CPU-friendly subset.
    Extract each zip; images from every archive go into ONE flat folder,
    e.g. C:\\datasets\\nih_chestxray\\images\\

    (Kaggle mirror, single download of everything, is also available at
    https://www.kaggle.com/datasets/nih-chest-xrays/data if you prefer —
    same Data_Entry_2017.csv + images/ layout.)

STEP 3 — Run this script:
    python scripts/prepare_nih_chestxray.py \
        --csv-path /path/to/Data_Entry_2017_v2020.csv \
        --images-dir /path/to/images \
        --max-images 3000
--------------------------------------------------------------------------

Like the FracAtlas script, this writes absolute image paths, so images are
never copied or duplicated.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# The 14 official NIH ChestX-ray14 finding labels (plus "No Finding" which
# we drop, since it isn't an anomaly class).
NIH_FINDINGS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia",
]


def main():
    parser = argparse.ArgumentParser(description="Prepare a local NIH ChestX-ray14 slice for training")
    parser.add_argument("--csv-path", required=True, help="Path to Data_Entry_2017(_v2020).csv")
    parser.add_argument("--images-dir", required=True, help="Folder containing the extracted PNG images (flat)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--max-images",
        type=int,
        default=3000,
        help="Cap on total images used — keep this modest on CPU-only machines (default: 3000)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=NIH_FINDINGS,
        help="Subset of NIH findings to keep as classes (default: all 14)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {args.csv_path}")
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    unknown = set(args.classes) - set(NIH_FINDINGS)
    if unknown:
        raise ValueError(f"Unknown NIH finding(s): {unknown}. Valid options: {NIH_FINDINGS}")

    df = pd.read_csv(args.csv_path)
    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError(
            f"Expected columns 'Image Index' and 'Finding Labels' not found. "
            f"Got: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows from metadata CSV.")

    # Multi-hot encode the pipe-separated Finding Labels column, e.g.
    # "Cardiomegaly|Effusion" -> {Cardiomegaly: 1, Effusion: 1, ...: 0}
    for finding in args.classes:
        df[finding] = df["Finding Labels"].apply(
            lambda s: int(finding in str(s).split("|"))
        )

    df["image_path"] = df["Image Index"].apply(lambda name: os.path.join(args.images_dir, name))

    # Only keep rows whose image file actually exists locally — this is what
    # lets the script work correctly even if you've only extracted 1-2 of
    # the 12 image archives instead of the full dataset.
    exists_mask = df["image_path"].apply(os.path.exists)
    found = exists_mask.sum()
    print(f"{found}/{len(df)} referenced images found in '{args.images_dir}'.")
    if found == 0:
        raise RuntimeError(
            "No images from the metadata CSV were found in --images-dir. "
            "Check that you extracted the NIH image zip(s) into that folder "
            "and that filenames weren't renamed during extraction."
        )
    df = df[exists_mask].reset_index(drop=True)

    if args.max_images and len(df) > args.max_images:
        # Prefer keeping a mix of positive and negative samples across
        # classes rather than a pure random slice, which for a dataset this
        # imbalanced could otherwise yield a subset with almost no positives.
        has_any_finding = df[args.classes].sum(axis=1) > 0
        positives = df[has_any_finding]
        negatives = df[~has_any_finding]

        n_pos = min(len(positives), args.max_images // 2)
        n_neg = min(len(negatives), args.max_images - n_pos)

        sampled = pd.concat(
            [
                positives.sample(n=n_pos, random_state=args.seed),
                negatives.sample(n=n_neg, random_state=args.seed),
            ]
        ).sample(frac=1, random_state=args.seed)  # shuffle
        df = sampled.reset_index(drop=True)
        print(
            f"Subsampled to {len(df)} images ({n_pos} with >=1 finding, "
            f"{n_neg} with no finding) for --max-images={args.max_images}"
        )

    final_cols = ["image_path"] + args.classes
    final_df = df[final_cols]

    # Stratifying multi-label data isn't directly supported by
    # train_test_split; approximate it with a "has any finding" flag so the
    # positive/negative ratio is at least preserved across the split.
    strat_key = (final_df[args.classes].sum(axis=1) > 0).astype(int)
    train_df, val_df = train_test_split(
        final_df, test_size=args.val_fraction, random_state=args.seed, stratify=strat_key
    )

    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)

    print(f"\nWrote {len(train_df)} training rows -> {config.TRAIN_CSV}")
    print(f"Wrote {len(val_df)} validation rows -> {config.VAL_CSV}")
    print("\nPer-class positive counts (train):")
    print(train_df[args.classes].sum().sort_values(ascending=False))

    print(
        "\nIMPORTANT: update config.py so CLASS_NAMES matches these classes:\n"
        f"    CLASS_NAMES = {args.classes!r}\n"
        "Then run: python train.py --epochs 10 --imbalance-strategy weighted_loss"
    )


if __name__ == "__main__":
    main()
