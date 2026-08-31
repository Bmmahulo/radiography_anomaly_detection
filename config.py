"""
config.py
---------
Central configuration for the Radiography Anomaly Detection prototype.
Keeping all paths / hyperparameters / constants in one place makes the
train / inference / API scripts consistent and easy to tune.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset_sample")
TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_labels.csv")
IMAGE_ROOT = os.path.join(DATA_DIR, "images")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")

STATIC_DIR = os.path.join(BASE_DIR, "static")
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")

LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in (CHECKPOINT_DIR, STATIC_DIR, HEATMAP_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Multi-label anomaly classes. Swap this list to match whichever dataset you
# point the pipeline at (e.g. NIH ChestX-ray14 has 14 findings, FracAtlas is
# effectively binary "fracture" / "no fracture").
CLASS_NAMES = ["fracture"]
NUM_CLASSES = len(CLASS_NAMES)

# Threshold above which a class is considered "positive" for reporting.
CLASS_THRESHOLD = 0.5

# A scan is flagged as HIGH PRIORITY if the max class probability exceeds this.
HIGH_PRIORITY_THRESHOLD = 0.75
MEDIUM_PRIORITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BACKBONE = os.environ.get("ANOMALY_BACKBONE", "efficientnet_b0")  # or "resnet50"
PRETRAINED = True
IMG_SIZE = 380 if "efficientnet" in BACKBONE else 224

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 5
SEED = 42

# Handles class imbalance: "weighted_loss" uses pos_weight in BCE,
# "sampler" uses a WeightedRandomSampler, "both" applies both strategies.
IMBALANCE_STRATEGY = os.environ.get("IMBALANCE_STRATEGY", "weighted_loss")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# ImageNet normalization stats (used since backbones are ImageNet-pretrained)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 8000))
MAX_UPLOAD_SIZE_MB = 25
ALLOWED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}
