# Architecture Experiment 05: U-Net++ (Nested U-Net)

## Overview
U-Net++ (Zhou et al., 2018) redesigns the skip pathways of standard U-Net by inserting nested dense convolutional sub-graphs between encoder and decoder feature maps. Dense skip connections reduce the semantic gap between encoder and decoder feature representations before fusion.

## Architectural Specification
- **Input Shape**: `[B, 1, 256, 256]` (Grayscale breast ultrasound images)
- **Output Shape**: `[B, 1, 256, 256]` (Binary segmentation logits)
- **Encoder Channels**: 64 -> 128 -> 256 -> 512
- **Bottleneck**: 1024 channels
- **Nested Pathways**: Dense grid of 10 sub-node convolution blocks ($x^{0,0}, x^{0,1}, x^{0,2}, x^{0,3}, x^{0,4}, x^{1,0}, x^{1,1}, x^{1,2}, x^{1,3}, x^{2,0}, x^{2,1}, x^{2,2}, x^{3,0}, x^{3,1}, x^{4,0}$).
- **Deep Supervision**: Disabled by default (`deep_supervision=False`) to preserve the common Dice Loss training protocol and single-logit output head.
- **Trainable Parameter Count**: **36,164,097 parameters** (+5,127,616 params relative to standard U-Net baseline).

## Research Protocol Compliance
- Uses exact master split: `train_70.csv` (440), `val_20.csv` (125), `test_10.csv` (63).
- **Test set is completely LOCKED and UNTOUCHED.**
- Training protocol: Adam optimizer, lr=1e-4, weight_decay=1e-4, Dice Loss, 100 epochs (when officially trained).
