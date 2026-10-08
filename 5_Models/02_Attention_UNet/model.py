"""
Attention U-Net Architecture Implementation (Oktay et al., 2018).
Second model architecture in the BUSI Phase-1 Segmentation Benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block:
    (Conv2D -> BatchNorm -> ReLU) * 2 with 3x3 kernel and padding=1.
    Matches standard U-Net baseline.
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


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).
    
    Filters encoder skip connections using gating signals from coarser decoder features.
    
    Mechanism:
    1. Take encoder skip feature map x (in_channels_x) as the attention feature.
    2. Take decoder/gating feature g (in_channels_g) as the gating signal.
    3. Project both into intermediate representation (inter_channels via 1x1 conv + BN).
    4. Combine them via element-wise addition.
    5. Apply ReLU non-linearity.
    6. Map to single-channel scalar attention coefficient via 1x1 conv + BN + Sigmoid.
    7. Multiply attention coefficients alpha element-wise with encoder skip feature map x.
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
        """
        Args:
            x: Encoder skip feature map [B, C_x, H, W]
            g: Decoder gating signal [B, C_g, H_g, W_g]
            
        Returns:
            Gated encoder skip feature map [B, C_x, H, W]
        """
        # Ensure spatial dimensions match (gating signal is upsampled if needed)
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        f = self.relu(g1 + x1)
        alpha = self.psi(f)
        
        return x * alpha


class AttentionUpBlock(nn.Module):
    """
    Decoder Block with Attention Gate:
    Upsample Decoder Feature -> Attention Gate on Skip Feature -> Concatenate -> DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.ag = AttentionGate(in_channels_x=out_channels, in_channels_g=in_channels, inter_channels=out_channels // 2)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            # Upsample converts in_channels to out_channels (e.g., 1024 -> 512)
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # Skip connection x2 has out_channels (e.g. 512), gating signal x1_up has out_channels (e.g. 512)
            self.ag = AttentionGate(in_channels_x=out_channels, in_channels_g=out_channels, inter_channels=out_channels // 2)
            # Concatenation of x2_gated (out_channels) and x1_up (out_channels) = out_channels * 2
            self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Decoder feature map from deeper stage [B, C_in, H1, W1]
            x2: Encoder skip connection feature map [B, C_out, H2, W2]
        """
        x1 = self.up(x1)
        
        # Spatial padding alignment if dimensions differ slightly by odd pixels
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Apply Attention Gate to skip feature x2 using decoder gating signal x1
        x2_gated = self.ag(x=x2, g=x1)

        # Concatenate along channel dimension
        x = torch.cat([x2_gated, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature maps to 1 output channel logit."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    2D Attention U-Net Architecture (Oktay et al., 2018).
    
    Architectural Specification:
    - Input Shape: [B, 1, 256, 256] (Grayscale ultrasound images)
    - Output Shape: [B, 1, 256, 256] (Binary segmentation logits)
    - Encoder Depth: 4 downsampling stages (5 resolution levels total)
    - Channel Expansion: 64 -> 128 -> 256 -> 512 -> 1024 (Bottleneck)
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
        self.inc = DoubleConv(in_channels, c)               # Level 0: 64
        self.down1 = DownBlock(c, c * 2)                     # Level 1: 128
        self.down2 = DownBlock(c * 2, c * 4)                 # Level 2: 256
        self.down3 = DownBlock(c * 4, c * 8)                 # Level 3: 512
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(c * 8, (c * 16) // factor)    # Bottleneck: 1024

        # Decoder with Attention Gates (Expanding Path)
        self.up1 = AttentionUpBlock(c * 16, (c * 8) // factor, bilinear) # 1024 -> 512
        self.up2 = AttentionUpBlock(c * 8, (c * 4) // factor, bilinear)  # 512 -> 256
        self.up3 = AttentionUpBlock(c * 4, (c * 2) // factor, bilinear)  # 256 -> 128
        self.up4 = AttentionUpBlock(c * 2, c, bilinear)                 # 128 -> 64

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
