"""
model.py — 2.5D U-Net with EfficientNet-B5 backbone

2.5D: input is [B, n_slices, H, W] — adjacent slices stacked as channels,
giving depth context without 3-D conv overhead.

Differential LR (FIX-4) is applied in train.py by accessing:
    model.model.encoder           → lr = 3e-5
    model.model.decoder           → lr = 3e-4
    model.model.segmentation_head → lr = 3e-4
These are standard smp.Unet sub-module names.

No functional changes from the previous version.
"""

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("Please install: pip install segmentation-models-pytorch")


class GITractUNet(nn.Module):
    """
    2.5D U-Net using EfficientNet-B5 encoder.

    Args:
        n_slices   : number of input channels (adjacent slices stacked)
        n_classes  : number of output segmentation classes
        encoder    : SMP encoder backbone name
        pretrained : use ImageNet pretrained weights
        img_size   : kept for API compatibility, unused internally
    """

    def __init__(
        self,
        n_slices:   int  = 3,
        n_classes:  int  = 3,
        encoder:    str  = "efficientnet-b5",
        pretrained: bool = True,
        img_size:   int  = 320,
    ):
        super().__init__()
        self.n_slices  = n_slices
        self.n_classes = n_classes

        # SMP handles in_channels != 3 by averaging pretrained weights
        # across the new channel dimension — no manual adaptation needed.
        self.model = smp.Unet(
            encoder_name    = encoder,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = n_slices,
            classes         = n_classes,
            activation      = None,   # raw logits; sigmoid applied in loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_slices, H, W]  →  logits: [B, n_classes, H, W]"""
        return self.model(x)


def build_model(n_slices:   int  = 3,
                n_classes:  int  = 3,
                pretrained: bool = True,
                device:     str  = "cuda") -> GITractUNet:
    """Factory — returns model on the specified device."""
    model = GITractUNet(
        n_slices   = n_slices,
        n_classes  = n_classes,
        encoder    = "efficientnet-b5",
        pretrained = pretrained,
    )
    return model.to(device)