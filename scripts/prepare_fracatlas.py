"""
scripts/prepare_fracatlas.py
------------------------------
Converts the real FracAtlas dataset into the CSV format expected by
data/dataset.py (image_path + one 0/1 column per class).

FracAtlas is single-label: every scan is either "fractured" or not, so this
script produces a single-class dataset. Update config.py accordingly:

    CLASS_NAMES = ["fracture"]

--------------------------------------------------------------------------
STEP 1 — Download (manual, ~323 MB, no registration required):
    https://doi.org/10.6084/m9.figshare.22363012
    Download the "FracAtlas.zip" file from that Figshare page.

STEP 2 — Extract it. You should end up with a folder structure like:
    FracAtlas/
      images/
        Fractured/        (IMG0000001.jpg, ...)
        Non_fractured/     (IMG0000002.jpg, ...)
      Annotations/
      utilities/
      dataset.csv

STEP 3 — Run this script pointing at that extracted folder:
    python scripts/prepare_fracatlas.py --source-dir /path/to/FracAtlas
--------------------------------------------------------------------------

The script does NOT copy the (potentially large) image files — it writes
CSVs with absolute image paths pointing directly at your extracted FracAtlas
folder, so `config.IMAGE_ROOT` is bypassed for this dataset (paths are
absolute, and data/dataset.py already handles absolute paths as-is).
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

EXPECTED_FRACTURED_COL_CANDIDATES = ["fractured", "Fractured"]
EXPECTED_ID_COL_CANDIDATES = ["image_id", "Image_Id", "image_name", "filename"]


def _find_column(df: pd.DataFrame, candidates, purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for '{purpose}' in dataset.csv. "
        f"Looked for {candidates}, found columns: {list(df.columns)}. "
        "FracAtlas's schema may have changed — open dataset.csv and check "
        "manually, then adjust this script's candidate lists."
    )


def build_image_path(row, image_col: str, fractured_col: str, images_dir: str) -> str:
    name = str(row[image_col]).strip()
    if not os.path.splitext(name)[1]:
        name += ".jpg"  # FracAtlas filenames are IMG####### .jpg

    subfolder = "Fractured" if int(row[fractured_col]) == 1 else "Non_fractured"
    return os.path.join(images_dir, subfolder, name)


def main():
    parser = argparse.ArgumentParser(description="Prepare FracAtlas for training")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to the extracted FracAtlas folder (contains dataset.csv and images/)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="Fraction of data held out for validation"
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on total images used (useful for a faster first pass on CPU)",
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.source_dir, "dataset.csv")
    images_dir = os.path.join(args.source_dir, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"dataset.csv not found at '{csv_path}'. Did you point --source-dir at the "
            "extracted FracAtlas folder (the one directly containing dataset.csv)?"
        )
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ folder not found at '{images_dir}'.")

    df = pd.read_csv(csv_path)
    id_col = _find_column(df, EXPECTED_ID_COL_CANDIDATES, "image identifier")
    fractured_col = _find_column(df, EXPECTED_FRACTURED_COL_CANDIDATES, "fractured label")

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Using id column='{id_col}', fractured column='{fractured_col}'")

    df["image_path"] = df.apply(
        lambda row: build_image_path(row, id_col, fractured_col, images_dir), axis=1
    )
    df["fracture"] = df[fractured_col].astype(int)

    # Drop any rows whose referenced image file doesn't actually exist on disk
    # (guards against a partial/corrupt extraction).
    exists_mask = df["image_path"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing:
        print(f"WARNING: {missing} rows reference image files that don't exist on disk — dropping them.")
    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            "No valid images found after filtering. Check that --source-dir points at a "
            "correctly extracted FracAtlas folder."
        )

    if args.max_images and len(df) > args.max_images:
        df = df.sample(n=args.max_images, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled down to {len(df)} images (--max-images={args.max_images})")

    final_df = df[["image_path", "fracture"]]

    train_df, val_df = train_test_split(
        final_df,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=final_df["fracture"],
    )

    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)

    print(f"\nWrote {len(train_df)} training rows -> {config.TRAIN_CSV}")
    print(f"Wrote {len(val_df)} validation rows -> {config.VAL_CSV}")
    print(f"Class balance (train): \n{train_df['fracture'].value_counts()}")

    print(
        "\nIMPORTANT: update config.py so CLASS_NAMES matches this single-label dataset:\n"
        '    CLASS_NAMES = ["fracture"]\n'
        "Then run: python train.py --epochs 10 --backbone resnet34"
    )


if __name__ == "__main__":
    main()
