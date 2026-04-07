"""
model.py — 2.5D U-Net with EfficientNet-B5 backbone
Based on segmentation-models-pytorch (smp)

2.5D extension: input is [B, n_slices, H, W] — stacked adjacent slices
as channels, giving the model depth context without 3D conv overhead.
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "Please install: pip install segmentation-models-pytorch"
    )


class GITractUNet(nn.Module):
    """
    2.5D U-Net using EfficientNet-B5 encoder.

    Args:
        n_slices    : number of input channels (adjacent slices stacked)
        n_classes   : number of output segmentation classes
        encoder     : encoder backbone name
        pretrained  : use ImageNet pretrained weights
        img_size    : input image size (used for weight adaptation)
    """

    def __init__(
        self,
        n_slices=3,
        n_classes=3,
        encoder="efficientnet-b5",
        pretrained=True,
        img_size=320,
    ):
        super().__init__()
        self.n_slices  = n_slices
        self.n_classes = n_classes

        encoder_weights = "imagenet" if pretrained else None

        # Pass in_channels directly — SMP handles the first-conv adaptation
        # internally when in_channels != 3, averaging pretrained weights across
        # the new channel dimension. No manual _adapt_first_conv needed.
        self.model = smp.Unet(
            encoder_name    = encoder,
            encoder_weights = encoder_weights,
            in_channels     = n_slices,
            classes         = n_classes,
            activation      = None,   # Raw logits — sigmoid applied in loss
        )

    def forward(self, x):
        """
        Args:
            x: [B, n_slices, H, W]
        Returns:
            logits: [B, n_classes, H, W]
        """
        return self.model(x)


def build_model(n_slices=3, n_classes=3, pretrained=True, device="cuda"):
    """Factory function — returns model on the specified device."""
    model = GITractUNet(
        n_slices   = n_slices,
        n_classes  = n_classes,
        encoder    = "efficientnet-b5",
        pretrained = pretrained,
    )
    return model.to(device)