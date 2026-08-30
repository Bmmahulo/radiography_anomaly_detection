"""
data/dataset.py
----------------
PyTorch Dataset for radiography multi-label anomaly classification.

Designed to be compatible with common public chest/skeletal X-ray datasets:
  - NIH ChestX-ray14: CSV of "Image Index" + pipe-separated "Finding Labels"
  - FracAtlas: CSV of image filename + binary "fractured" column
  - Any custom dataset described by a CSV with one column per class name in
    config.CLASS_NAMES (0/1 values) plus an "image_path" column.

Reads PNG/JPEG via OpenCV and DICOM (.dcm) via pydicom, normalizing both to
an 8-bit 3-channel array so the same pretrained ImageNet backbone can be
applied to either source.
"""

import os
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import CLASS_NAMES, IMAGE_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pydicom

    _PYDICOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDICOM_AVAILABLE = False


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk regardless of whether it is DICOM, PNG, or JPEG,
    and returns an 8-bit 3-channel (H, W, 3) RGB numpy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        if not _PYDICOM_AVAILABLE:
            raise RuntimeError(
                "pydicom is not installed but a .dcm file was provided. "
                "Run `pip install pydicom`."
            )
        try:
            dcm = pydicom.dcmread(path)
            arr = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM file '{path}': {e}") from e

        # Normalize to 0-255 8-bit range.
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        arr = (arr * 255.0).astype(np.uint8)

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return arr

    # PNG / JPEG / other OpenCV-readable formats
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(
            f"OpenCV failed to decode image '{path}'. File may be corrupt "
            "or in an unsupported format."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class RadiographyDataset(Dataset):
    """
    Expects a CSV with columns:
        image_path, <class_1>, <class_2>, ..., <class_N>
    where each class column contains a 0/1 label.

    `image_path` may be absolute or relative to `image_root`.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str = IMAGE_ROOT,
        class_names=None,
        transform: Optional[Callable] = None,
    ):
        self.class_names = class_names or CLASS_NAMES
        self.image_root = image_root
        self.transform = transform

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Labels CSV not found at '{csv_path}'. See README for the "
                "expected dataset layout, or run scripts/make_sample_dataset.py."
            )

        self.df = pd.read_csv(csv_path)

        missing_cols = [c for c in self.class_names if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV '{csv_path}' is missing expected label columns: {missing_cols}. "
                f"Expected columns: image_path + {self.class_names}"
            )
        if "image_path" not in self.df.columns:
            raise ValueError(f"CSV '{csv_path}' must contain an 'image_path' column.")

        logger.info(
            "Loaded dataset from %s: %d samples, %d classes",
            csv_path,
            len(self.df),
            len(self.class_names),
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, image_path: str) -> str:
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(self.image_root, image_path)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._resolve_path(row["image_path"])

        try:
            image = load_image(path)
        except Exception as e:
            logger.error("Error loading sample %d (%s): %s", idx, path, e)
            # Fail loudly rather than silently corrupting a training batch with
            # a black image — the caller (DataLoader) should surface this.
            raise

        labels = row[self.class_names].values.astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, torch.from_numpy(labels)

    def get_class_pos_counts(self) -> np.ndarray:
        """Positive-sample counts per class, used for imbalance handling."""
        return self.df[self.class_names].sum(axis=0).values

    def get_sample_weights(self) -> np.ndarray:
        """
        Per-sample weight for WeightedRandomSampler: samples containing any
        rare positive finding are up-weighted so the model sees enough of them
        each epoch despite chest X-ray anomaly datasets typically being
        dominated by "normal" scans.
        """
        pos_counts = self.get_class_pos_counts()
        # Avoid divide-by-zero for classes with 0 positives in this split.
        class_weights = 1.0 / np.clip(pos_counts, 1, None)

        labels = self.df[self.class_names].values
        # A sample's weight is the max weight across its positive classes;
        # samples with no positive findings get a baseline weight of 1.
        sample_weights = np.where(
            labels.sum(axis=1, keepdims=True) > 0,
            (labels * class_weights).max(axis=1, keepdims=True),
            1.0,
        ).flatten()
        return sample_weights
