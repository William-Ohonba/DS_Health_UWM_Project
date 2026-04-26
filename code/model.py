"""
model.py — 2.5D U-Net with EfficientNet-B5 backbone

2.5D: input is [B, n_slices, H, W] — adjacent slices stacked as channels,
giving depth context without 3-D conv overhead.

Differential LR (FIX-4) is applied in train.py by accessing:
    model.model.encoder           → lr = 1e-4   [FIX-23]
    model.model.decoder           → lr = 3e-4   (3× encoder, not 10×)
    model.model.segmentation_head → lr = 3e-4

FIXES applied:
  [FIX-27] Dropout(0.2) injected before the segmentation head.
  [FIX-30] Auxiliary presence detection head added.
  [FIX-31] presence_head now operates on DETACHED encoder features.
           Previously presence gradients flowed back through the full
           encoder (2048-channel deepest feature map), contributing ~30%
           of total encoder gradient signal and actively competing with
           segmentation gradients during the critical first 10–15 epochs
           when encoder features are forming. This caused the encoder to
           oscillate between "segment" and "classify presence" objectives,
           producing the loss spikes and dice collapses observed at epochs
           14, 19, 21, 29. Fix: features[-1].detach() stops all presence
           gradients at the encoder boundary. The presence head still
           trains (its own parameters receive gradients), but it no longer
           corrupts encoder feature learning.
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("Please install: pip install segmentation-models-pytorch")


class GITractUNet(nn.Module):
    """
    2.5D U-Net using EfficientNet-B5 encoder + auxiliary presence head.

    Args:
        n_slices   : number of input channels (adjacent slices stacked)
        n_classes  : number of output segmentation classes
        encoder    : SMP encoder backbone name
        pretrained : use ImageNet pretrained weights
        img_size   : kept for API compatibility, unused internally
        dropout    : dropout probability before segmentation head [FIX-27]

    Returns (from forward):
        seg_logits      : [B, n_classes, H, W]  raw segmentation logits
        presence_logits : [B, n_classes]         per-class presence logits [FIX-30]
    """

    def __init__(
        self,
        n_slices:   int   = 3,
        n_classes:  int   = 3,
        encoder:    str   = "efficientnet-b5",
        pretrained: bool  = True,
        img_size:   int   = 320,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.n_slices  = n_slices
        self.n_classes = n_classes

        self.model = smp.Unet(
            encoder_name    = encoder,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = n_slices,
            classes         = n_classes,
            activation      = None,
        )

        # [FIX-27] Wrap seg head with Dropout2d
        self.model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            self.model.segmentation_head,
        )

        # [FIX-30] Presence detection head.
        # [FIX-31] Operates on detached features — see module docstring.
        enc_out_ch = self.model.encoder.out_channels[-1]
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, C_enc, 1, 1]
            nn.Flatten(),              # [B, C_enc]
            nn.Linear(enc_out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_classes), # [B, n_classes]
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, n_slices, H, W]
        Returns:
            seg_logits      [B, n_classes, H, W]
            presence_logits [B, n_classes]
        """
        features = self.model.encoder(x)

        # [FIX-31] Detach before presence head so its gradients cannot
        # corrupt encoder weights. The presence head trains on its own
        # parameters only; the encoder is supervised solely by seg loss.
        presence_logits = self.presence_head(features[-1].detach())

        decoder_out = self.model.decoder(features)
        seg_logits  = self.model.segmentation_head(decoder_out)
        return seg_logits, presence_logits


def build_model(n_slices:   int   = 3,
                n_classes:  int   = 3,
                pretrained: bool  = True,
                device:     str   = "cuda",
                dropout:    float = 0.2) -> GITractUNet:
    """Factory — returns model on the specified device."""
    model = GITractUNet(
        n_slices   = n_slices,
        n_classes  = n_classes,
        encoder    = "efficientnet-b5",
        pretrained = pretrained,
        dropout    = dropout,
    )
    return model.to(device)