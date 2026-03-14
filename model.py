"""
model.py — Dental AI
U-Net Architecture for multi-class dental segmentation
4 classes: Background, Caries, Infection, Restoration
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks (the core U-Net unit)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv + MaxPool (returns both for skip connection)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Upsample + concatenate skip connection + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)   # in_channels because of concat

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch (can happen with odd-sized inputs)
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)   # concat along channel dim
        return self.conv(x)


# ─────────────────────────────────────────────
#  U-NET
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for dental X-ray segmentation.

    Input : [B, 3, 512, 512]  — RGB X-ray
    Output: [B, 4, 512, 512]  — logits per class (Background/Caries/Infection/Restoration)
    """

    def __init__(self, in_channels=3, num_classes=4, features=[64, 128, 256, 512]):
        super().__init__()

        # ── Encoder (Contracting Path) ──
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # ── Bottleneck ──
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder (Expanding Path) ──
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f))

        # ── Final 1×1 conv → class logits ──
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)   # [B, num_classes, H, W]


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model  = UNet(in_channels=3, num_classes=4)
    x      = torch.randn(2, 3, 512, 512)    # batch of 2 images
    output = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {output.shape}")   # should be [2, 4, 512, 512]

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
