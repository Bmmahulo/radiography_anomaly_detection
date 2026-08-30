"""
inference.py
-------------
Runs anomaly-detection inference on a single scan image and saves an
annotated image with a Grad-CAM heatmap overlay for the highest-scoring
(most concerning) finding.

CLI usage:
    python inference.py --image /path/to/scan.png
    python inference.py --image /path/to/scan.dcm --checkpoint checkpoints/best_model.pt

Can also be imported and called programmatically (this is exactly what
api/routes.py does for the FastAPI endpoint).
"""

import argparse
import json
import os
import uuid
from typing import Dict

import cv2
import numpy as np
import torch

import config
from data.dataset import load_image
from data.transforms import get_inference_transforms
from models.classifier import build_model
from models.gradcam import GradCAM, overlay_heatmap_on_image
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache so the API server doesn't reload the model on every request.
_MODEL_CACHE = {}


def get_model(checkpoint_path: str = config.BEST_MODEL_PATH):
    """Loads (and caches) the classifier for repeated inference calls."""
    key = checkpoint_path
    if key not in _MODEL_CACHE:
        logger.info("Loading model for inference from %s", checkpoint_path)
        model = build_model(
            backbone_name=config.BACKBONE,
            num_classes=config.NUM_CLASSES,
            pretrained=False,  # weights come from checkpoint, not ImageNet, if available
            checkpoint_path=checkpoint_path,
            device=config.DEVICE,
        )
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def determine_priority(max_prob: float) -> str:
    if max_prob >= config.HIGH_PRIORITY_THRESHOLD:
        return "HIGH"
    if max_prob >= config.MEDIUM_PRIORITY_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def run_inference(
    image_path: str,
    checkpoint_path: str = config.BEST_MODEL_PATH,
    save_heatmap: bool = True,
    heatmap_dir: str = config.HEATMAP_DIR,
) -> Dict:
    """
    Runs the full inference pipeline on one image:
      1. Load + preprocess image
      2. Forward pass -> per-class probabilities
      3. Grad-CAM on the top-scoring class
      4. Save annotated heatmap overlay to disk

    Returns a JSON-serializable dict describing the result.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = get_model(checkpoint_path)
    transform = get_inference_transforms()

    # --- Load & preprocess -------------------------------------------------
    try:
        raw_image = load_image(image_path)  # (H, W, 3) uint8 RGB
    except Exception as e:
        logger.error("Failed to load image '%s': %s", image_path, e)
        raise

    transformed = transform(image=raw_image)
    input_tensor = transformed["image"].unsqueeze(0).to(config.DEVICE)  # (1, C, H, W)

    # --- Forward pass --------------------------------------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (num_classes,)

    class_scores = {
        name: float(prob) for name, prob in zip(config.CLASS_NAMES, probs)
    }
    top_class_idx = int(np.argmax(probs))
    top_class_name = config.CLASS_NAMES[top_class_idx]
    top_score = float(probs[top_class_idx])

    flagged_classes = [
        name for name, p in class_scores.items() if p >= config.CLASS_THRESHOLD
    ]

    priority = determine_priority(top_score)

    result = {
        "image_path": image_path,
        "priority": priority,
        "top_finding": top_class_name,
        "top_finding_confidence": round(top_score, 4),
        "flagged_findings": flagged_classes,
        "class_scores": {k: round(v, 4) for k, v in class_scores.items()},
        "heatmap_url": None,
    }

    # --- Grad-CAM ------------------------------------------------------------
    if save_heatmap:
        try:
            gradcam = GradCAM(model, model.get_target_layer())
            cam = gradcam.generate(input_tensor, class_idx=top_class_idx)
            gradcam.remove_hooks()

            # Resize the *original* image to the model's input resolution so
            # the heatmap aligns with visible anatomical structures.
            resized_original = cv2.resize(
                raw_image, (config.IMG_SIZE, config.IMG_SIZE)
            )
            overlay = overlay_heatmap_on_image(resized_original, cam)

            os.makedirs(heatmap_dir, exist_ok=True)
            filename = f"heatmap_{uuid.uuid4().hex[:12]}.png"
            out_path = os.path.join(heatmap_dir, filename)

            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(out_path, overlay_bgr)
            if not success:
                raise RuntimeError(f"cv2.imwrite failed for path: {out_path}")

            result["heatmap_url"] = f"/static/heatmaps/{filename}"
            logger.info("Saved Grad-CAM heatmap to %s", out_path)
        except Exception as e:
            # Heatmap generation failing should not take down the whole
            # prediction — the classification result is still useful without it.
            logger.error("Grad-CAM generation failed: %s", e)
            result["heatmap_error"] = str(e)

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Run anomaly detection inference on a scan")
    p.add_argument("--image", required=True, help="path to a PNG/JPEG/DICOM scan")
    p.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        result = run_inference(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            save_heatmap=not args.no_heatmap,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
