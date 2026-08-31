"""
scripts/validate_images.py
----------------------------
Scans the images referenced in one or more dataset CSVs, detects corrupted /
truncated files (the "Premature end of JPEG file" warnings you see during
training come from exactly this), and optionally removes the offending rows
so a training run isn't quietly learning from partially-garbage images.

Why this matters: OpenCV's JPEG decoder (used by data/dataset.py at train
time) often does NOT raise a Python exception on a truncated file — it just
prints a C-level warning to stderr and returns whatever partial pixel data
it managed to decode. Training keeps running, but some of your images may
be corrupted, near-black, or partially blank without anything ever failing
loudly. This script uses PIL's stricter checking (which DOES raise on
truncation) to properly identify the affected files.

Usage:
    # Report only — doesn't modify anything
    python scripts/validate_images.py

    # Report AND write cleaned CSVs (originals backed up as *.bak)
    python scripts/validate_images.py --clean

    # Check specific CSV(s) instead of the default train/val pair
    python scripts/validate_images.py --csv path/to/some_labels.csv --clean
"""

import argparse
import os
import shutil
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

try:
    import pydicom

    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False


def check_image(path: str) -> str:
    """
    Returns an empty string if the image is fine, otherwise a short reason
    string describing why it was flagged as invalid.
    """
    if not os.path.exists(path):
        return "file not found"

    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        if not _PYDICOM_AVAILABLE:
            return "pydicom not installed — cannot validate .dcm files"
        try:
            dcm = pydicom.dcmread(path)
            _ = dcm.pixel_array  # forces full pixel data decode
            return ""
        except Exception as e:
            return f"DICOM read error: {e}"

    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check
        # verify() closes the file handle and doesn't always catch truncation
        # during full decode, so re-open and force a full load as well.
        with Image.open(path) as img:
            img.load()
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def validate_csv(csv_path: str, do_clean: bool) -> None:
    if not os.path.exists(csv_path):
        print(f"SKIP: '{csv_path}' does not exist.")
        return

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        print(f"SKIP: '{csv_path}' has no 'image_path' column.")
        return

    print(f"\nValidating {len(df)} images referenced in {csv_path} ...")

    bad_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
        reason = check_image(row["image_path"])
        if reason:
            bad_rows.append((idx, row["image_path"], reason))

    if not bad_rows:
        print(f"All {len(df)} images OK in {csv_path}.")
        return

    print(f"\nFound {len(bad_rows)}/{len(df)} corrupted/unreadable image(s):")
    for _, path, reason in bad_rows[:25]:  # cap printed detail to keep output readable
        print(f"  - {path}\n      -> {reason}")
    if len(bad_rows) > 25:
        print(f"  ... and {len(bad_rows) - 25} more.")

    if do_clean:
        bad_indices = [idx for idx, _, _ in bad_rows]
        cleaned_df = df.drop(index=bad_indices).reset_index(drop=True)

        backup_path = csv_path + ".bak"
        shutil.copy2(csv_path, backup_path)
        cleaned_df.to_csv(csv_path, index=False)

        print(
            f"\nRemoved {len(bad_rows)} row(s). "
            f"Original backed up to '{backup_path}'. "
            f"Cleaned CSV written to '{csv_path}' ({len(cleaned_df)} rows remain)."
        )
    else:
        print(
            "\nRe-run with --clean to remove these rows and write a cleaned CSV "
            "(original will be backed up as *.bak)."
        )


def main():
    parser = argparse.ArgumentParser(description="Validate images referenced in dataset CSVs")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[config.TRAIN_CSV, config.VAL_CSV],
        help="CSV file(s) to validate (default: current train + val CSVs)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Write cleaned CSVs with bad rows removed (originals backed up as *.bak)",
    )
    args = parser.parse_args()

    for csv_path in args.csv:
        validate_csv(csv_path, args.clean)

    print("\nDone.")


if __name__ == "__main__":
    main()
