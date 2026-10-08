# Loss Function Benchmark Report: Attention U-Net

## 1. Study Overview
- **Objective:** Controlled evaluation of loss functions on the selected optimal architecture (**Attention U-Net**).
- **Architecture Specification:** Attention U-Net (Oktay et al., 2018), 31,387,049 parameters.
- **Dataset:** BUSI Phase-1 Single-Mask Cohort (440 Train / 125 Validation / 63 Test).
- **Test Set Status:** **LOCKED & UNTOUCHED (0 test samples loaded or evaluated). All comparisons use validation set only.**
- **Training Protocol:** 100 Epochs, Adam optimizer (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau, Batch Size 16, Seed 42. Early stopping DISABLED.

## 2. Validation Metrics Comparison Table

| Loss Function | Best Epoch | Dice | IoU | Precision | Recall | Specificity | HD95 (px) [Valid N] | NSD @ 1px | NSD @ 2px | NSD @ 3px |
|---|---|---|---|---|---|---|---|---|---|---|
| `Dice_baseline` | N/A | **0.7527 ± 0.2642** | 0.6602 ± 0.2730 | 0.8152 ± 0.2461 | 0.7666 ± 0.2905 | 0.9834 ± 0.0319 | 31.47 ± 36.24 (121/125) | 0.3969 ± 0.2770 | 0.5196 ± 0.3188 | 0.5972 ± 0.3263 |
| `BCE` | 93 | **0.7415 ± 0.2821** | 0.6528 ± 0.2882 | 0.8124 ± 0.2640 | 0.7438 ± 0.3011 | 0.9883 ± 0.0225 | 36.90 ± 37.25 (121/125) | 0.3916 ± 0.2582 | 0.5191 ± 0.3082 | 0.5958 ± 0.3217 |
| `BCE_Dice` **(Best Val Dice)** | 97 | **0.7572 ± 0.2575** | 0.6650 ± 0.2730 | 0.8188 ± 0.2250 | 0.7734 ± 0.2830 | 0.9855 ± 0.0246 | 34.16 ± 39.91 (124/125) | 0.3957 ± 0.2554 | 0.5270 ± 0.3005 | 0.6102 ± 0.3103 |
| `Focal` | 66 | **0.6787 ± 0.3149** | 0.5860 ± 0.3076 | 0.7344 ± 0.3141 | 0.7102 ± 0.3371 | 0.9827 ± 0.0299 | 37.85 ± 39.15 (117/125) | 0.3026 ± 0.2367 | 0.4206 ± 0.2985 | 0.5012 ± 0.3254 |
| `Tversky` | 98 | **0.7282 ± 0.2623** | 0.6264 ± 0.2650 | 0.7337 ± 0.2559 | 0.8101 ± 0.2826 | 0.9751 ± 0.0362 | 47.44 ± 51.97 (122/125) | 0.3044 ± 0.2035 | 0.4503 ± 0.2651 | 0.5544 ± 0.2901 |

## 3. Loss Formulations & Parameters

1. **Dice (Baseline):**
   - Formulation: Binary Soft Dice Loss
   - Parameters: `smooth=1e-6`

2. **Binary Cross-Entropy (BCE):**
   - Formulation: Binary Cross-Entropy with Logits
   - Parameters: `from_logits=True`, `reduction=mean`

3. **Combined BCE + Dice (BCE_Dice):**
   - Formulation: BCE + Dice (bce_weight * BCE + dice_weight * Dice)
   - Parameters: `bce_weight=1.0`, `dice_weight=1.0`, `smooth=1e-6`

4. **Binary Focal Loss (Focal):**
   - Formulation: Binary Focal Loss (gamma=2.0, alpha=0.25)
   - Parameters: `alpha=0.25`, `gamma=2.0`, `from_logits=True`

5. **Tversky Loss (Tversky):**
   - Formulation: Tversky Index Loss (alpha=0.7, beta=0.3)
   - Parameters: `alpha=0.7`, `beta=0.3`, `smooth=1e-6`

## 4. Key Scientific Observations
- **Top Performer by Validation Dice:** `BCE_Dice` achieved validation Dice of **0.7572 ± 0.2575** (Best Epoch 97).
- Complete per-image predictions, learning curves, and model checkpoints are archived in their respective experiment folders.
