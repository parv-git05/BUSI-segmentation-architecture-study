"""
U-Net++ (Nested U-Net) Architecture Implementation (Zhou et al., 2018).
Fifth model architecture in the BUSI Phase-1 Segmentation Benchmark.
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


class UpConv(nn.Module):
    """Upsampling block using 2x2 ConvTranspose2d."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature maps to 1 output channel logit."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    2D U-Net++ (Nested U-Net) Architecture (Zhou et al., 2018).
    
    Architectural Specification:
    - Input Shape: [B, 1, 256, 256] (Grayscale ultrasound images)
    - Output Shape: [B, 1, 256, 256] (Binary segmentation logits)
    - Encoder Channels: 64 -> 128 -> 256 -> 512
    - Bottleneck Channels: 1024
    - Skip Pathways: Dense nested skip pathways bridging encoder and decoder sub-graphs.
    - Deep Supervision: Disabled by default (deep_supervision=False) to preserve the common Dice Loss protocol.
    - Output Head: Final 1x1 conv from top node x^{0,4}.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        deep_supervision: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.deep_supervision = deep_supervision

        c = [base_channels * (2 ** i) for i in range(5)]  # [64, 128, 256, 512, 1024]

        # Upsamplers
        self.up1_0 = UpConv(c[1], c[0])
        self.up2_0 = UpConv(c[2], c[1])
        self.up3_0 = UpConv(c[3], c[2])
        self.up4_0 = UpConv(c[4], c[3])

        self.up1_1 = UpConv(c[1], c[0])
        self.up2_1 = UpConv(c[2], c[1])
        self.up3_1 = UpConv(c[3], c[2])

        self.up1_2 = UpConv(c[1], c[0])
        self.up2_2 = UpConv(c[2], c[1])

        self.up1_3 = UpConv(c[1], c[0])

        # Grid Nodes DoubleConvs
        # Level 0 (c[0] = 64)
        self.conv0_0 = DoubleConv(in_channels, c[0])
        self.conv0_1 = DoubleConv(c[0] * 2, c[0])
        self.conv0_2 = DoubleConv(c[0] * 3, c[0])
        self.conv0_3 = DoubleConv(c[0] * 4, c[0])
        self.conv0_4 = DoubleConv(c[0] * 5, c[0])

        # Level 1 (c[1] = 128)
        self.conv1_0 = DownBlock(c[0], c[1])
        self.conv1_1 = DoubleConv(c[1] * 2, c[1])
        self.conv1_2 = DoubleConv(c[1] * 3, c[1])
        self.conv1_3 = DoubleConv(c[1] * 4, c[1])

        # Level 2 (c[2] = 256)
        self.conv2_0 = DownBlock(c[1], c[2])
        self.conv2_1 = DoubleConv(c[2] * 2, c[2])
        self.conv2_2 = DoubleConv(c[2] * 3, c[2])

        # Level 3 (c[3] = 512)
        self.conv3_0 = DownBlock(c[2], c[3])
        self.conv3_1 = DoubleConv(c[3] * 2, c[3])

        # Level 4 (c[4] = 1024)
        self.conv4_0 = DownBlock(c[3], c[4])

        # Final Output Layer(s)
        self.final = OutConv(c[0], out_channels)

        if self.deep_supervision:
            self.final1 = OutConv(c[0], out_channels)
            self.final2 = OutConv(c[0], out_channels)
            self.final3 = OutConv(c[0], out_channels)

    def _pad_and_cat(self, tensors: list, upsampled_tensor: torch.Tensor) -> torch.Tensor:
        """Aligns spatial dimensions of upsampled tensor with target skip tensors before concatenation."""
        target_shape = tensors[0].size()[2:]
        if upsampled_tensor.size()[2:] != target_shape:
            diff_y = target_shape[0] - upsampled_tensor.size()[2]
            diff_x = target_shape[1] - upsampled_tensor.size()[3]
            upsampled_tensor = F.pad(upsampled_tensor, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return torch.cat(tensors + [upsampled_tensor], dim=1)

    def forward(self, x: torch.Tensor):
        # Column 0 (Encoder Backbone)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        # Column 1
        x0_1 = self.conv0_1(self._pad_and_cat([x0_0], self.up1_0(x1_0)))
        x1_1 = self.conv1_1(self._pad_and_cat([x1_0], self.up2_0(x2_0)))
        x2_1 = self.conv2_1(self._pad_and_cat([x2_0], self.up3_0(x3_0)))
        x3_1 = self.conv3_1(self._pad_and_cat([x3_0], self.up4_0(x4_0)))

        # Column 2
        x0_2 = self.conv0_2(self._pad_and_cat([x0_0, x0_1], self.up1_1(x1_1)))
        x1_2 = self.conv1_2(self._pad_and_cat([x1_0, x1_1], self.up2_1(x2_1)))
        x2_2 = self.conv2_2(self._pad_and_cat([x2_0, x2_1], self.up3_1(x3_1)))

        # Column 3
        x0_3 = self.conv0_3(self._pad_and_cat([x0_0, x0_1, x0_2], self.up1_2(x1_2)))
        x1_3 = self.conv1_3(self._pad_and_cat([x1_0, x1_1, x1_2], self.up2_2(x2_2)))

        # Column 4
        x0_4 = self.conv0_4(self._pad_and_cat([x0_0, x0_1, x0_2, x0_3], self.up1_3(x1_3)))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final(x0_4)
            return [output1, output2, output3, output4]

        output = self.final(x0_4)
        return output

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
