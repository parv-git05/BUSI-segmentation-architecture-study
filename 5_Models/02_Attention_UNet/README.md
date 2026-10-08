# Architecture Experiment 02: Attention U-Net

## Overview
Attention U-Net (Oktay et al., 2018) is the second model architecture evaluated in the BUSI Lesion Segmentation Architecture Study. It incorporates Additive Attention Gates (AGs) into the standard U-Net skip connections to dynamically filter features passed from the encoder to the decoder.

## Architectural Specification
- **Input Shape**: `[B, 1, 256, 256]` (Grayscale breast ultrasound images)
- **Output Shape**: `[B, 1, 256, 256]` (Binary segmentation logits)
- **Encoder Depth**: 4 downsampling stages (64 -> 128 -> 256 -> 512)
- **Bottleneck**: 1024 channels
- **Decoder Depth**: 4 upsampling stages (512 -> 256 -> 128 -> 64)
- **Attention Gates**: 4 Additive Attention Gates placed on encoder skip connections prior to concatenation with decoder feature maps.
- **Trainable Parameter Count**: **31,387,049 parameters** (+350,568 params relative to standard U-Net baseline).

## Attention Gate Mechanism
For an encoder skip feature map $x \in \mathbb{R}^{F_x \times H \times W}$ and decoder gating signal $g \in \mathbb{R}^{F_g \times H \times W}$:
1. Intermediate Projection:
   $$g_1 = W_g(g), \quad x_1 = W_x(x)$$
   where $W_g \in \mathbb{R}^{F_{int} \times F_g \times 1 \times 1}$ and $W_x \in \mathbb{R}^{F_{int} \times F_x \times 1 \times 1}$ are 1x1 convolutions followed by 2D Batch Normalization.
2. Additive Combination & Non-Linearity:
   $$f = \text{ReLU}(g_1 + x_1)$$
3. Attention Coefficient Computation:
   $$\alpha = \sigma(\psi(f))$$
   where $\psi \in \mathbb{R}^{1 \times F_{int} \times 1 \times 1}$ is a 1x1 convolution with Batch Normalization and $\sigma$ is Sigmoid activation.
4. Feature Gating:
   $$\hat{x} = x \odot \alpha$$

## Research Protocol Compliance
- Uses exact master split: `train_70.csv` (440), `val_20.csv` (125), `test_10.csv` (63).
- **Test set is completely LOCKED and UNTOUCHED.**
- Training protocol: Adam optimizer, lr=1e-4, weight_decay=1e-4, Dice Loss, 100 epochs (when officially trained).
