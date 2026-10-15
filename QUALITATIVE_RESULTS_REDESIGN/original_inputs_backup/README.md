# Qualitative Results Redesign Report

This folder contains the complete, publication-quality redesign of qualitative segmentation figures for the BUSI breast ultrasound lesion segmentation research paper.

---

## 1. Executive Summary & Figure Inventory

| Figure File | Format | DPI / Size | Section | Candidate Option | Samples Used |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `architecture_comparison_option_A.png` | PNG | 300 DPI | Architecture Comparison | **Option A** | `benign (273)`, `benign (99)`, `malignant (183)` |
| `architecture_comparison_option_A.pdf` | Vector PDF | Publication | Architecture Comparison | **Option A** | (Vector export for LaTeX integration) |
| `architecture_comparison_option_B.png` | PNG | 300 DPI | Architecture Comparison | **Option B** | `benign (275)`, `benign (149)`, `malignant (26)` |
| `architecture_comparison_option_B.pdf` | Vector PDF | Publication | Architecture Comparison | **Option B** | (Vector export for LaTeX integration) |
| `loss_comparison_option_A.png` | PNG | 300 DPI | Loss-Function Comparison | **Option A** | `benign (142)`, `benign (38)`, `malignant (85)` |
| `loss_comparison_option_A.pdf` | Vector PDF | Publication | Loss-Function Comparison | **Option A** | (Vector export for LaTeX integration) |
| `loss_comparison_option_B.png` | PNG | 300 DPI | Loss-Function Comparison | **Option B** | `benign (82)`, `benign (297)`, `malignant (12)` |
| `loss_comparison_option_B.pdf` | Vector PDF | Publication | Loss-Function Comparison | **Option B** | (Vector export for LaTeX integration) |
| `contact_sheet_preview.png` | JPEG/PNG | Overview | All Figures | Contact Sheet | 2x2 grid preview of all four options |
| `qualitative_sample_selection.csv` | CSV | Manifest | All Figures | Complete Audit | Full sample metadata, paths, and per-model metrics |

---

## 2. Structural & Layout Specification

Each figure implements the 6-column layout:
- **Column 1:** `Ground Truth (GT)`
  - *Top Row:* Single-mask ground-truth binary lesion annotation (256x256).
  - *Bottom Row:* Original grayscale B-mode ultrasound image with Case ID and Class badge (Green for Benign, Vermilion for Malignant).
- **Columns 2–6:** The five comparative conditions:
  - *Architecture Section:* 1. U-Net, 2. Attention U-Net, 3. Residual U-Net, 4. Attention Residual U-Net, 5. U-Net++.
  - *Loss Function Section:* 1. Dice Loss, 2. BCE Loss, 3. BCE + Dice Loss (Final Model), 4. Focal Loss, 5. Tversky Loss.
- **Top Row of Columns 2–6:** Exactly **five identical copies of the same GT mask** for that sample, positioned directly above each model's prediction to facilitate vertical eye-tracking and immediate contour comparison.
- **Bottom Row of Columns 2–6:** High-fidelity semi-transparent prediction overlays over the original ultrasound image.
- **Three Sample Groups:** Each figure contains exactly three sample groups (one group per vertical block), with 2 benign cases and 1 malignant case, ensuring representative clinical diversity.

---

## 3. Overlay Conventions & Color Palette

- **Ultrasound Background:** Grayscale scan scaled to [0.0, 1.0].
- **Predicted Lesion Mask:** Translucent high-visibility Cyan (`#00E5FF`, opacity $\alpha = 0.38$) with a solid Cyan contour (`#00E5FF`, width = 1.8 px).
- **Ground Truth Boundary:** Crisp Lime Green contour line (`#00E676`, width = 2.2 px).
- **Spatial Concordance:** Inside the green boundary, the cyan shaded area represents True Positive region overlap. Cyan outside green denotes False Positives; unshaded ultrasound tissue inside green denotes False Negatives.
- **Quantitative Sub-label:** Underneath each overlay panel, exact per-image **Dice Similarity Coefficient (DSC)** and **Intersection over Union (IoU)** are displayed.

---

## 4. Checkpoints & Reproducibility Traceability

### Section 1: Architecture Comparison Checkpoints
All models trained under the frozen baseline Dice loss protocol:
1. **U-Net:** `01_UNet/experiments/EXP-ARCH-01_UNET/checkpoints/best_model.pth` (Epoch 93, 31,036,481 params)
2. **Attention U-Net:** `02_Attention_UNet/experiments/EXP-ARCH-02_ATTENTION_UNET/checkpoints/best_model.pth` (Epoch 96, 31,387,049 params)
3. **Residual U-Net:** `03_Residual_UNet/experiments/EXP-ARCH-03_RESIDUAL_UNET/checkpoints/best_model.pth` (Epoch 89, 32,435,073 params)
4. **Attention Residual U-Net:** `04_Attention_Residual_UNet/experiments/EXP-ARCH-04_ATTENTION_RESIDUAL_UNET/checkpoints/best_model.pth` (Epoch 93, 32,785,641 params)
5. **U-Net++:** `05_UNetPP/experiments/EXP-ARCH-05_UNETPP/checkpoints/best_model.pth` (Epoch 88, 36,164,097 params)

### Section 2: Loss-Function Comparison Checkpoints
All models utilize the Attention U-Net architecture (31,387,049 params):
1. **Dice Baseline:** `Best_Architecture/Dice_baseline/checkpoints/best_model.pth` (Epoch 96)
2. **BCE:** `Best_Architecture/BCE/checkpoints/best_model.pth` (Epoch 93)
3. **BCE + Dice (Final Selected Checkpoint):** `Best_Architecture/BCE_Dice/checkpoints/best_model.pth` (Epoch 97)
4. **Focal:** `Best_Architecture/Focal/checkpoints/best_model.pth` (Epoch 66)
5. **Tversky:** `Best_Architecture/Tversky/checkpoints/best_model.pth` (Epoch 98)

---

## 5. Visual Quality Assessment & Recommendation

- **Architecture Comparison Recommendation:** **Option A** (`architecture_comparison_option_A.png`) is visually strongest. It clearly demonstrates Attention U-Net's sharp boundary delineations on `benign (273)` and `benign (99)`, while highlighting the comparative strengths across models on the challenging irregular margins of `malignant (183)`.
- **Loss Comparison Recommendation:** **Option A** (`loss_comparison_option_A.png`) is visually strongest. It clearly illustrates how the combined BCE + Dice loss stabilizes boundary confidence compared to BCE-alone or Focal loss on `benign (142)` and `malignant (85)`.
