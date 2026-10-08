# Architecture Experiment 03: Residual U-Net

## Overview
Residual U-Net incorporates residual connections into the standard 2D U-Net encoder and decoder blocks. By establishing shortcut paths across dual 3x3 convolution layers, residual connections mitigate vanishing gradient issues and facilitate feature propagation.

## Architectural Specification
- **Input Shape**: `[B, 1, 256, 256]` (Grayscale breast ultrasound images)
- **Output Shape**: `[B, 1, 256, 256]` (Binary segmentation logits)
- **Encoder Depth**: 4 downsampling stages (64 -> 128 -> 256 -> 512)
- **Bottleneck**: 1024 channels
- **Decoder Depth**: 4 upsampling stages (512 -> 256 -> 128 -> 64)
- **Residual Blocks**: Each stage uses a `ResBlock` with:
  - Main Path: `Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN`
  - Shortcut Path: `Identity()` if $C_{in} == C_{out}$, else `Conv1x1 -> BN`
  - Output: `ReLU(Main(x) + Shortcut(x))`
- **Trainable Parameter Count**: **32,435,073 parameters** (+1,398,592 params relative to standard U-Net baseline).

## Research Protocol Compliance
- Uses exact master split: `train_70.csv` (440), `val_20.csv` (125), `test_10.csv` (63).
- **Test set is completely LOCKED and UNTOUCHED.**
- Training protocol: Adam optimizer, lr=1e-4, weight_decay=1e-4, Dice Loss, 100 epochs (when officially trained).
