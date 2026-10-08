"""
Attention Residual U-Net Architecture Implementation.
Fourth model architecture in the BUSI Phase-1 Segmentation Benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual Convolutional Block:
    Main path: (Conv2d 3x3 -> BatchNorm2d -> ReLU -> Conv2d 3x3 -> BatchNorm2d)
    Shortcut path: Conv2d 1x1 + BatchNorm2d (if in_channels != out_channels) else Identity.
    Output: ReLU(Main(x) + Shortcut(x))
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class DownResBlock(nn.Module):
    """Downscaling block: MaxPool2d (2x2, stride 2) followed by ResBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_res = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_res(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).
    Filters encoder skip connections using gating signals from coarser decoder features.
    """
    def __init__(self, in_channels_x: int, in_channels_g: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels_x // 2
            if inter_channels == 0:
                inter_channels = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        f = self.relu(g1 + x1)
        alpha = self.psi(f)
        
        return x * alpha


class AttentionUpResBlock(nn.Module):
    """
    Decoder Block combining Upsampling, Attention Gate, Concatenation, and ResBlock.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.ag = AttentionGate(in_channels_x=out_channels, in_channels_g=in_channels, inter_channels=out_channels // 2)
            self.conv = ResBlock(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.ag = AttentionGate(in_channels_x=out_channels, in_channels_g=out_channels, inter_channels=out_channels // 2)
            self.conv = ResBlock(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Apply Attention Gate to encoder skip feature x2 using decoder gating signal x1
        x2_gated = self.ag(x=x2, g=x1)

        x = torch.cat([x2_gated, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature maps to 1 output channel logit."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionResidualUNet(nn.Module):
    """
    2D Attention Residual U-Net Architecture.
    
    Architectural Specification:
    - Input Shape: [B, 1, 256, 256] (Grayscale ultrasound images)
    - Output Shape: [B, 1, 256, 256] (Binary segmentation logits)
    - Encoder Depth: 4 downsampling stages (5 resolution levels total)
    - Channel Expansion: 64 -> 128 -> 256 -> 512 -> 1024 (Bottleneck)
    - Convolution Blocks: ResBlock with identity/projection shortcut connections
    - Downsampling: MaxPool2d (2x2, stride=2)
    - Upsampling: ConvTranspose2d (2x2, stride=2)
    - Attention Mechanism: Additive Attention Gates on all skip connections
    - Normalization: Batch Normalization (2D)
    - Activation: ReLU
    - Output Layer: 1x1 Conv (raw unnormalized logits)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        c = base_channels  # 64

        # Encoder (Contracting Path)
        self.inc = ResBlock(in_channels, c)                   # Level 0: 64
        self.down1 = DownResBlock(c, c * 2)                   # Level 1: 128
        self.down2 = DownResBlock(c * 2, c * 4)               # Level 2: 256
        self.down3 = DownResBlock(c * 4, c * 8)               # Level 3: 512
        factor = 2 if bilinear else 1
        self.down4 = DownResBlock(c * 8, (c * 16) // factor)  # Bottleneck: 1024

        # Decoder with Attention Gates and Residual Blocks (Expanding Path)
        self.up1 = AttentionUpResBlock(c * 16, (c * 8) // factor, bilinear)
        self.up2 = AttentionUpResBlock(c * 8, (c * 4) // factor, bilinear)
        self.up3 = AttentionUpResBlock(c * 4, (c * 2) // factor, bilinear)
        self.up4 = AttentionUpResBlock(c * 2, c, bilinear)

        # Output Layer
        self.outc = OutConv(c, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
