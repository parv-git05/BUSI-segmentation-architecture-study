"""
Standard U-Net Architecture Implementation (Ronneberger et al., 2015).
First model architecture in the BUSI Phase-1 Segmentation Benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block:
    (Conv2D -> BatchNorm -> ReLU) * 2 with 3x3 kernel and padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling block: MaxPool2d (2x2, stride 2) followed by DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling block: Upsample (ConvTranspose2d or Bilinear) -> Concatenate Skip Connection -> DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle spatial padding if features differ by odd pixel dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature maps to 1 output channel logit."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 2D U-Net Architecture.
    
    Architectural Specification:
    - Input Shape: [B, 1, 256, 256] (Grayscale ultrasound images)
    - Output Shape: [B, 1, 256, 256] (Binary segmentation logits)
    - Encoder Depth: 4 downsampling stages (5 resolution levels total)
    - Channel Expansion: 64 -> 128 -> 256 -> 512 -> 1024 (Bottleneck)
    - Downsampling: MaxPool2d (2x2, stride=2)
    - Upsampling: ConvTranspose2d (2x2, stride=2)
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
        self.inc = DoubleConv(in_channels, c)         # Level 0: 64
        self.down1 = DownBlock(c, c * 2)               # Level 1: 128
        self.down2 = DownBlock(c * 2, c * 4)           # Level 2: 256
        self.down3 = DownBlock(c * 4, c * 8)           # Level 3: 512
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(c * 8, (c * 16) // factor)  # Bottleneck: 1024

        # Decoder (Expanding Path)
        self.up1 = UpBlock(c * 16, (c * 8) // factor, bilinear)
        self.up2 = UpBlock(c * 8, (c * 4) // factor, bilinear)
        self.up3 = UpBlock(c * 4, (c * 2) // factor, bilinear)
        self.up4 = UpBlock(c * 2, c, bilinear)

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
