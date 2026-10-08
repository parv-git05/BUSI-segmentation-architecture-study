# Architecture Experiment 04: Attention Residual U-Net

## Overview
Attention Residual U-Net combines residual convolutional blocks with additive attention gates. Residual connections inside encoder/decoder stages enhance intra-stage gradient flow, while attention gates on skip connections selectively weight inter-stage feature transfer.

## Architectural Specification
- **Input Shape**: `[B, 1, 256, 256]` (Grayscale breast ultrasound images)
- **Output Shape**: `[B, 1, 256, 256]` (Binary segmentation logits)
- **Encoder Depth**: 4 downsampling stages (64 -> 128 -> 256 -> 512)
- **Bottleneck**: 1024 channels
- **Decoder Depth**: 4 upsampling stages (512 -> 256 -> 128 -> 64)
- **Convolution Blocks**: `ResBlock` with 3x3 dual convolutions and identity/projection shortcuts.
- **Attention Gates**: 4 `AttentionGate` modules on skip connections.
- **Trainable Parameter Count**: **32,785,641 parameters** (+1,749,160 params relative to standard U-Net baseline).

## Research Protocol Compliance
- Uses exact master split: `train_70.csv` (440), `val_20.csv` (125), `test_10.csv` (63).
- **Test set is completely LOCKED and UNTOUCHED.**
- Training protocol: Adam optimizer, lr=1e-4, weight_decay=1e-4, Dice Loss, 100 epochs (when officially trained).
