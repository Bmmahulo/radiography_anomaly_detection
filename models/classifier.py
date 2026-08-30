"""
models/classifier.py
---------------------
Multi-label anomaly classifier built on a pretrained torchvision backbone
(EfficientNet-B0 or ResNet-50 by default). The final classification layer
is replaced with a linear head producing one logit per anomaly class;
sigmoid (applied outside the model, in the loss/inference code) turns each
logit into an independent probability, since a scan can exhibit more than
one finding simultaneously.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

from utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_BACKBONES = {
    "efficientnet_b0",
    "efficientnet_b3",
    "resnet50",
    "resnet34",
}


class AnomalyClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone_name not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from {sorted(_SUPPORTED_BACKBONES)}."
            )

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone, feature_dim, self.target_layer_name = self._build_backbone(
            backbone_name, pretrained
        )

        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

        logger.info(
            "Initialized AnomalyClassifier: backbone=%s, num_classes=%d, "
            "pretrained=%s, feature_dim=%d",
            backbone_name,
            num_classes,
            pretrained,
            feature_dim,
        )

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, str]:
        """
        Returns (backbone_module, feature_dim, target_layer_name).
        target_layer_name is the module path Grad-CAM should hook into
        (the last convolutional block before global pooling).
        """
        if name == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = tv_models.efficientnet_b0(weights=weights)
            feature_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feature_dim, "features.8"

        if name == "efficientnet_b3":
            weights = tv_models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            net = tv_models.efficientnet_b3(weights=weights)
            feature_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feature_dim, "features.8"

        if name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet50(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feature_dim, "layer4"

        if name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet34(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feature_dim, "layer4"

        raise ValueError(f"Unhandled backbone '{name}'")  # unreachable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits

    def get_target_layer(self) -> nn.Module:
        """Resolves the dotted target_layer_name into the actual nn.Module,
        used by GradCAM to register forward/backward hooks."""
        module = self.backbone
        for part in self.target_layer_name.split("."):
            module = getattr(module, part) if not part.isdigit() else module[int(part)]
        return module


def build_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: str = None,
    device: torch.device = None,
) -> AnomalyClassifier:
    """Convenience factory used by train.py / inference.py / app.py."""
    model = AnomalyClassifier(
        backbone_name=backbone_name, num_classes=num_classes, pretrained=pretrained
    )

    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location=device or "cpu")
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            model.load_state_dict(state_dict)
            logger.info("Loaded model weights from checkpoint: %s", checkpoint_path)
        except FileNotFoundError:
            logger.warning(
                "Checkpoint '%s' not found — using randomly-initialized/"
                "ImageNet-pretrained weights only. Train the model first "
                "for meaningful predictions.",
                checkpoint_path,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}': {e}"
            ) from e

    if device is not None:
        model = model.to(device)

    return model
