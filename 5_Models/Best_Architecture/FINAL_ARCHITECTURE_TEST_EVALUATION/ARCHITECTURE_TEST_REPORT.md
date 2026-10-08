# Final Architecture Test Evaluation Report: BUSI Segmentation Benchmark

---

## 1. Study Objective
This report provides the final, unbiased held-out test evaluation comparing all five deep learning segmentation architectures benchmarked under the common baseline Dice loss protocol. The evaluated models represent the official architecture benchmark cohort:
1. Standard U-Net
2. Attention U-Net
3. Residual U-Net
4. Attention Residual U-Net
5. U-Net++

Each architecture was trained using the identical Dice Loss objective on the 70% training cohort (`train_70.csv`, N=440), checkpointed at its optimal validation-Dice epoch on the 20% validation cohort (`val_20.csv`, N=125), and evaluated deterministically here on the locked 10% test set (`test_10.csv`, N=63).

---

## 2. Test Dataset & Isolation
- **Dataset:** BUSI Phase-1 Single-Mask Cohort
- **Manifest:** `5_Models/MASTER_SPLIT/test_10.csv`
- **Total Test Samples:** 63 images
- **Class Breakdown:**
  - **Benign:** 42 images (66.67%)
  - **Malignant:** 21 images (33.33%)
- **Test Isolation Statement:** The test set was locked and untouched throughout model design, hyperparameter selection, loss benchmarking, and checkpoint selection. All five models are evaluated once on the identical test cohort.

---

## 3. Experimental Controls (Frozen Across All Five Architectures)
- **Loss Function:** Dice Loss (`DiceLoss`, `smooth=1e-6`, `from_logits=True`)
- **Optimizer During Training:** Adam (`lr=1e-4`, `weight_decay=1e-4`, `batch_size=16`, `epochs=100`)
- **Preprocessing:**
  - Color mode: 1-channel Grayscale
  - Spatial resolution: 256 × 256 pixels (Image: Bilinear, Mask: Nearest Neighbor)
  - Value range: $[0.0, 1.0]$
  - Augmentation: None (strictly deterministic inference)
- **Inference Threshold:** Fixed at **0.5** applied to sigmoid probabilities: `Prediction = (sigmoid(logits) >= 0.5)`.

---

## 4. Models and Checkpoints Evaluated
| Architecture | Checkpoint File | Best Epoch | Trainable Parameters |
| :--- | :--- | :---: | :---: |
| **U-Net** | `01_UNet/experiments/EXP-ARCH-01_UNET/checkpoints/best_model.pth` | 93 | 31,036,481 |
| **Attention U-Net** | `02_Attention_UNet/experiments/EXP-ARCH-02_ATTENTION_UNET/checkpoints/best_model.pth` | 96 | 31,387,049 |
| **Residual U-Net** | `03_Residual_UNet/experiments/EXP-ARCH-03_RESIDUAL_UNET/checkpoints/best_model.pth` | 89 | 32,435,073 |
| **Attention Residual U-Net** | `04_Attention_Residual_UNet/experiments/EXP-ARCH-04_ATTENTION_RESIDUAL_UNET/checkpoints/best_model.pth` | 93 | 32,785,641 |
| **U-Net++** | `05_UNetPP/experiments/EXP-ARCH-05_UNETPP/checkpoints/best_model.pth` | 88 | 36,164,097 |

---

## 5. Consolidated Test Performance (N=63)
Macro-averaged per-image metrics across all 63 test images:

| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px) (Mean ± SD) | HD95 Valid N | NSD @ 1px | NSD @ 2px | NSD @ 3px |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | **0.6706 ± 0.3140** | 0.5767 ± 0.3104 | 0.7129 ± 0.3382 | 0.7168 ± 0.3212 | 0.9815 ± 0.0313 | 42.46 ± 44.25 | 59/63 | 0.2736 | 0.3925 | 0.4906 |
| **Attention U-Net** | **0.7069 ± 0.2978** | 0.6128 ± 0.2915 | 0.7483 ± 0.3050 | 0.7340 ± 0.3044 | 0.9846 ± 0.0274 | 42.40 ± 42.34 | 59/63 | 0.3157 | 0.4394 | 0.5284 |
| **Residual U-Net** | **0.7111 ± 0.3003** | 0.6199 ± 0.2973 | 0.7485 ± 0.3049 | 0.7412 ± 0.3118 | 0.9850 ± 0.0266 | 34.14 ± 36.19 | 58/63 | 0.3164 | 0.4487 | 0.5470 |
| **Attention Residual U-Net** | **0.6949 ± 0.3150** | 0.6061 ± 0.3094 | 0.7866 ± 0.3039 | 0.6804 ± 0.3398 | 0.9924 ± 0.0129 | 35.00 ± 39.14 | 57/63 | 0.3341 | 0.4645 | 0.5452 |
| **U-Net++** | **0.7066 ± 0.2971** | 0.6139 ± 0.2984 | 0.7588 ± 0.3209 | 0.7333 ± 0.2964 | 0.9850 ± 0.0289 | 34.76 ± 37.63 | 59/63 | 0.3264 | 0.4594 | 0.5440 |

---

## 6. Per-Class Test Performance Breakdown

### A. Benign Lesions (N=42)
| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px, Valid N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | 0.6827 ± 0.3420 | 0.6025 ± 0.3360 | 0.6720 ± 0.3637 | 0.7673 ± 0.3254 | 0.9815 ± 0.0338 | 43.11 px (39/42) |
| **Attention U-Net** | 0.7153 ± 0.3160 | 0.6304 ± 0.3107 | 0.7321 ± 0.3230 | 0.7603 ± 0.3199 | 0.9874 ± 0.0285 | 44.33 px (39/42) |
| **Residual U-Net** | 0.7401 ± 0.3052 | 0.6577 ± 0.3016 | 0.7342 ± 0.3171 | 0.7930 ± 0.3025 | 0.9871 ± 0.0275 | 30.15 px (38/42) |
| **Attention Residual U-Net** | 0.7234 ± 0.3305 | 0.6463 ± 0.3186 | 0.7658 ± 0.3238 | 0.7216 ± 0.3519 | 0.9944 ± 0.0085 | 27.27 px (37/42) |
| **U-Net++** | 0.7248 ± 0.3162 | 0.6440 ± 0.3161 | 0.7305 ± 0.3371 | 0.7901 ± 0.2936 | 0.9843 ± 0.0328 | 30.93 px (39/42) |

### B. Malignant Lesions (N=21)
| Architecture | Dice (Mean ± SD) | IoU (Mean ± SD) | Precision (Mean ± SD) | Recall (Mean ± SD) | Specificity (Mean ± SD) | HD95 (px, Valid N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | 0.6462 ± 0.2645 | 0.5250 ± 0.2604 | 0.7947 ± 0.2802 | 0.6159 ± 0.3032 | 0.9815 ± 0.0275 | 41.19 px (20/21) |
| **Attention U-Net** | 0.6900 ± 0.2725 | 0.5776 ± 0.2605 | 0.7809 ± 0.2782 | 0.6815 ± 0.2788 | 0.9789 ± 0.0254 | 38.65 px (20/21) |
| **Residual U-Net** | 0.6531 ± 0.2963 | 0.5444 ± 0.2879 | 0.7771 ± 0.2923 | 0.6377 ± 0.3191 | 0.9807 ± 0.0253 | 41.74 px (20/21) |
| **Attention Residual U-Net** | 0.6377 ± 0.2891 | 0.5256 ± 0.2886 | 0.8282 ± 0.2707 | 0.5979 ± 0.3149 | 0.9884 ± 0.0187 | 49.31 px (20/21) |
| **U-Net++** | 0.6702 ± 0.2662 | 0.5537 ± 0.2646 | 0.8156 ± 0.2939 | 0.6196 ± 0.2823 | 0.9864 ± 0.0205 | 42.22 px (20/21) |

---

## 7. Edge-Case Handling & Invalid Metric Observations
- **HD95 Edge Cases:** Under Euclidean distance transform definitions, if either the predicted foreground or the ground truth lesion contains zero pixels, the Hausdorff distance surface contour is mathematically undefined. Consistent with project standards, such cases are assigned `NaN` and excluded from `hd95_mean` without imputing zeros. The exact number of valid cases is reported in the table above.
- **Surface Distances (NSD):** Empty set pairs yield 1.0 (perfect empty match); single empty sets yield 0.0, ensuring 63/63 valid samples for all models.

---

## 8. Note on Loss Comparison Separation
> **Important Scope Note:**
> This evaluation strictly compares the five architectures trained under the baseline **Dice Loss** to isolate architectural impact.
> The separate loss study (evaluating BCE, Focal, Tversky, and BCE + Dice on Attention U-Net) is preserved in `5_Models/Best_Architecture/BCE_Dice/` and `5_Models/Best_Architecture/FINAL_TEST_EVALUATION/`.
