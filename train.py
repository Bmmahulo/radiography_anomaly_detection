"""
train.py
--------
Training entrypoint for the radiography anomaly classifier.

Usage:
    python train.py --epochs 25 --batch-size 16 --backbone efficientnet_b0
    python train.py --train-csv data/train.csv --val-csv data/val.csv

Handles:
    - Dataset loading (train/val split via separate CSVs)
    - Data augmentation (via data/transforms.py, Albumentations)
    - Class imbalance (pos_weight in BCEWithLogitsLoss and/or a
      WeightedRandomSampler — see config.IMBALANCE_STRATEGY)
    - Training loop with mixed precision, LR scheduling, early stopping
    - Evaluation each epoch: AUROC / Precision / Recall / F1 (utils/metrics.py)
    - Checkpointing best + last model
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config
from data.dataset import RadiographyDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.classifier import build_model
from utils.logger import get_logger
from utils.metrics import compute_metrics, format_metrics_report

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train the radiography anomaly classifier")
    p.add_argument("--train-csv", default=config.TRAIN_CSV)
    p.add_argument("--val-csv", default=config.VAL_CSV)
    p.add_argument("--image-root", default=config.IMAGE_ROOT)
    p.add_argument("--backbone", default=config.BACKBONE, choices=[
        "efficientnet_b0", "efficientnet_b3", "resnet50", "resnet34",
    ])
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--imbalance-strategy", default=config.IMBALANCE_STRATEGY,
                    choices=["none", "weighted_loss", "sampler", "both"])
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def build_dataloaders(args):
    train_ds = RadiographyDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=get_train_transforms(),
    )
    val_ds = RadiographyDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=get_val_transforms(),
    )

    sampler = None
    shuffle = True
    if args.imbalance_strategy in ("sampler", "both"):
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        shuffle = False  # mutually exclusive with sampler
        logger.info("Using WeightedRandomSampler for class-imbalance handling.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, val_ds, train_loader, val_loader


def compute_pos_weight(train_ds: RadiographyDataset) -> torch.Tensor:
    """pos_weight for BCEWithLogitsLoss = (num_negatives / num_positives) per class,
    which up-weights the loss contribution of rare positive findings."""
    pos_counts = train_ds.get_class_pos_counts()
    total = len(train_ds)
    neg_counts = total - pos_counts
    pos_weight = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(y_true, y_prob, class_names)
    return val_loss, metrics


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = config.DEVICE
    logger.info("Using device: %s", device)

    try:
        train_ds, val_ds, train_loader, val_loader = build_dataloaders(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to build datasets: %s", e)
        raise SystemExit(1) from e

    model = build_model(
        backbone_name=args.backbone,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        device=device,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Resumed model weights from %s", args.resume)

    pos_weight = None
    if args.imbalance_strategy in ("weighted_loss", "both"):
        pos_weight = compute_pos_weight(train_ds).to(device)
        logger.info("Using pos_weight for BCE loss: %s", pos_weight.tolist())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, metrics = evaluate(
            model, val_loader, criterion, device, config.CLASS_NAMES
        )
        scheduler.step(val_loss)
        elapsed = time.time() - start

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f "
            "macro_auroc=%.4f macro_recall=%.4f | %.1fs",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            metrics["macro"]["macro_auroc"],
            metrics["macro"]["macro_recall"],
            elapsed,
        )
        logger.info(format_metrics_report(metrics))

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "class_names": config.CLASS_NAMES,
                "val_loss": val_loss,
                "metrics": metrics["macro"],
            },
            config.LAST_MODEL_PATH,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "class_names": config.CLASS_NAMES,
                    "val_loss": val_loss,
                    "metrics": metrics["macro"],
                },
                config.BEST_MODEL_PATH,
            )
            logger.info(
                "New best model saved to %s (val_loss=%.4f)",
                config.BEST_MODEL_PATH,
                val_loss,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d epoch(s) (best val_loss=%.4f)",
                epochs_without_improvement,
                best_val_loss,
            )
            if epochs_without_improvement >= args.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    args.patience,
                )
                break

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
