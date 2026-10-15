# Qualitative Results Redesign Report (Revised)

This folder contains the complete, publication-quality revision of qualitative segmentation figures for the BUSI breast ultrasound lesion segmentation research paper.

---

## 1. Executive Summary & Figure Inventory

| Figure File | Format | DPI / Size | Section | Candidate Option | Samples Retained | Sample Count |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| `architecture_comparison_option_A.png` | PNG | 300 DPI | Architecture Comparison | **Option A** | `benign (273)`, `malignant (183)` | **2 samples** |
| `architecture_comparison_option_A.pdf` | Vector PDF | Publication | Architecture Comparison | **Option A** | (Vector export for LaTeX integration) | 2 samples |
| `architecture_comparison_option_B.png` | PNG | 300 DPI | Architecture Comparison | **Option B** | `benign (275)`, `benign (149)`, `malignant (26)` | 3 samples |
| `architecture_comparison_option_B.pdf` | Vector PDF | Publication | Architecture Comparison | **Option B** | (Vector export for LaTeX integration) | 3 samples |
| `loss_comparison_option_A.png` | PNG | 300 DPI | Loss-Function Comparison | **Option A** | `benign (38)`, `malignant (85)` | **2 samples** |
| `loss_comparison_option_A.pdf` | Vector PDF | Publication | Loss-Function Comparison | **Option A** | (Vector export for LaTeX integration) | 2 samples |
| `loss_comparison_option_B.png` | PNG | 300 DPI | Loss-Function Comparison | **Option B** | `benign (82)`, `benign (297)`, `malignant (12)` | 3 samples |
| `loss_comparison_option_B.pdf` | Vector PDF | Publication | Loss-Function Comparison | **Option B** | (Vector export for LaTeX integration) | 3 samples |
| `contact_sheet_preview.png` | JPEG/PNG | Overview | All Figures | Contact Sheet | 2x2 grid preview of all four revised options | - |
| `qualitative_sample_selection.csv` | CSV | Manifest | All Figures | Complete Audit | Full sample metadata, paths, and per-model metrics | 10 entries |

---

## 2. Revision Details & Key Improvements

1. **Repeated GT Mask Panels Cleaned**:
   - The watermark text `GT Reference` has been completely eliminated from columns 2–6.
   - Ground truth masks in columns 2–6 are pure, unannotated, pixel-identical binary masks aligned with the respective model/loss prediction.
   - Column 1 retains the clean label badge `Group 1: GT Mask`, `Group 2: GT Mask`, etc.

2. **Clean Original Ultrasound Panels**:
   - Filenames such as `benign (273).png` or `malignant (183).png` have been removed entirely from the ultrasound images.
   - The ultrasound images are unobstructed.
   - A clean class badge (`Benign` in dark green `#15803D`, `Malignant` in vermilion `#C2410C`) is displayed outside the image directly beneath each panel.

3. **Sample Reductions**:
   - **Architecture Comparison Option A**: Reduced to **exactly 2 samples** (`benign (273).png` and `malignant (183).png`), with `benign (99).png` removed.
   - **Loss-Function Comparison Option A**: Reduced to **exactly 2 samples** (`benign (38).png` and `malignant (85).png`), with `benign (142).png` removed.
   - **Option B Figures**: Preserved with their established 3 samples each for comprehensive secondary review.

4. **Prediction Boundary & Color Palette Coordinated**:
   - **Ground-Truth Boundary:** Retained bright lime green contour line (`#00E676`, width = 2.2 px).
   - **Predicted Lesion Boundary:** Updated from cyan to vibrant coral red (`#E53935`, width = 2.0 px), ensuring high visual contrast against the green GT contour.
   - **Predicted Mask Fill:** Translucent muted red tint (`#FF5252`, opacity $\alpha = 0.36$), harmonized with the red contour.
   - **Legend:** Fully updated with Green GT boundary, Coral Red predicted boundary, and Translucent red lesion region.

5. **Panel Sizing & High Sharpness**:
   - For 2-sample figures, canvas height is optimized (`18.0" x 11.2"`) so individual image panels are substantially larger and sharp at 300 DPI (approx 5,400 x 3,360 pixels) without squishing or excessive whitespace.
   - 3-sample figures use an expanded `18.0" x 15.6"` canvas at 300 DPI.
   - Vector PDFs exported alongside high-resolution PNGs for IEEE journal submission.

---

## 3. Structural & Layout Specification

Each figure implements the standardized 6-column layout:
- **Column 1:** `Ground Truth (GT)`
  - *Top Row:* Single-mask ground-truth binary lesion annotation (256x256) with Group label.
  - *Bottom Row:* Clean original grayscale B-mode ultrasound image with `Benign`/`Malignant` badge positioned beneath.
- **Columns 2–6:** The five comparative conditions:
  - *Architecture Section:* 1. U-Net, 2. Attention U-Net, 3. Residual U-Net, 4. Attention Residual U-Net, 5. U-Net++.
  - *Loss Function Section:* 1. Dice Loss, 2. BCE Loss, 3. BCE + Dice Loss (Final Model), 4. Focal Loss, 5. Tversky Loss.
- **Top Row of Columns 2–6:** Exactly five identical copies of the same GT mask for that sample, positioned directly above each model's prediction.
- **Bottom Row of Columns 2–6:** High-fidelity semi-transparent red prediction overlays over the original ultrasound image with DSC and IoU reported above each panel.

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

- **Architecture Comparison Recommendation:** **Option A** (`architecture_comparison_option_A.png`) is the primary recommendation. With 2 samples (`benign (273)` and `malignant (183)`), the panels are exceptionally large and clear. It highlights Attention U-Net's high boundary precision (Dice 0.9615 on benign 273) and robust delineation of the complex infiltrative margin on malignant 183 (Dice 0.9363).
- **Loss Comparison Recommendation:** **Option A** (`loss_comparison_option_A.png`) is the primary recommendation. With 2 samples (`benign (38)` and `malignant (85)`), it clearly demonstrates how the combined BCE + Dice loss achieves superior boundary concordance (Dice 0.9650 on benign 38, Dice 0.9445 on malignant 85) compared to single-objective BCE or Focal loss formulations.
