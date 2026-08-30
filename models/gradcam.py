"""
models/gradcam.py
------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Produces a heatmap over the input image highlighting the spatial regions
that most influenced the model's prediction for a given class — this is
the visual-explainability piece that lets a radiologist sanity-check *why*
a scan was flagged, rather than trusting an opaque confidence score.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self._bwd_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int
    ) -> np.ndarray:
        """
        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor, requires_grad
                          not necessary — we only need gradients w.r.t. activations.
            class_idx: index of the target class in the model's output layer.

        Returns:
            (H, W) float32 heatmap normalized to [0, 1], at the input tensor's
            spatial resolution.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)  # (1, num_classes)

        if class_idx < 0 or class_idx >= logits.shape[1]:
            raise ValueError(
                f"class_idx={class_idx} out of range for model with "
                f"{logits.shape[1]} output classes."
            )

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "GradCAM hooks did not capture activations/gradients. "
                "Verify the target_layer is actually used in the forward pass."
            )

        # Global-average-pool the gradients over spatial dims -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]; guard against a degenerate all-zero CAM.
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            logger.warning("Grad-CAM produced a near-constant activation map.")
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def overlay_heatmap_on_image(
    original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """
    Args:
        original_image: (H, W, 3) uint8 RGB image (the ORIGINAL, un-normalized
                         image, resized to the same resolution the CAM was
                         generated at).
        cam: (H, W) float32 heatmap in [0, 1].
        alpha: blend strength of the heatmap over the original image.

    Returns:
        (H, W, 3) uint8 RGB annotated image.
    """
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if original_image.shape[:2] != cam.shape[:2]:
        original_image = cv2.resize(
            original_image, (cam.shape[1], cam.shape[0])
        )

    overlay = cv2.addWeighted(
        original_image.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0
    )
    return overlay
