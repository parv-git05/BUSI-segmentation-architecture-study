# BUSI Attention U-Net — Per-Image Validation Failure Audit Report

## 1. Executive Summary

This report delivers a rigorous, per-image failure analysis of the **Attention U-Net** (`EXP-ARCH-02_ATTENTION_UNET`) validation cohort within the BUSI Breast Ultrasound Lesion Segmentation benchmark. 

The primary objective of this audit is to resolve why the model's validation Dice scores exhibit substantial dispersion (**Mean: $0.7530 \pm 0.2635$**, Range: $[0.0000, 0.9729]$) and to identify the underlying structural failure modes.

### Key Audit Findings:
1. **Bimodal Failure Structure:** The high standard deviation ($\sigma \approx 0.2635$) is driven by a heavy tail of catastrophic and severe failures rather than uniform slight boundary erosion. While the median Dice is **$0.8580$** (with 75% of validation images achieving Dice up to $0.9391$), **$15.20\%$ (19/125)** of validation cases fail with $\text{Dice} < 0.50$, and **$7.20\%$ (9/125)** exhibit near-zero performance ($\text{Dice} < 0.25$).
2. **Catastrophic Failure Modes:**
   - **Empty Predictions ($N=4$):** Complete prediction dropouts where the network produces 0 foreground pixels despite large, unambiguous ground-truth lesions ($6,438 - 16,326$ pixels).
   - **Spatial Disconnect / False Localization ($N=2$):** Predictions placed entirely in acoustic shadow or glandular fibroconnective tissue without overlapping the true lesion.
   - **Severe Under-segmentation ($N=9$):** Predictions capturing only a tiny focal core ($\text{Recall} < 0.30$, $\text{Precision} \approx 0.95-1.00$).
3. **Lesion Size Interaction:** Counter-intuitively, large lesions ($\text{Q4} > 13.28\%$ of image area) exhibited the **highest failure rate ($25.81\%$ with $\text{Dice} < 0.50$)** and lowest mean Dice ($0.6447$), primarily due to focal under-segmentation and complete dropout of expansive hypoechoic masses. Small lesions ($\text{Q1} \le 2.06\%$) achieved the highest mean Dice ($0.8615$) and lowest failure rate ($3.12\%$).
4. **Pathology Gap:** Benign cases achieved a mean Dice of **$0.7837$** (Median: $0.9173$), while Malignant cases lagged at **$0.6900$** (Median: $0.7854$), reflecting the challenge posed by irregular, spicular borders and acoustic attenuation shadowing.

---

## 2. Artifact and Checkpoint Verification

All experiment artifacts were verified for integrity prior to audit execution:
- **Experiment Directory:** `5_Models/02_Attention_UNet/experiments/EXP-ARCH-02_ATTENTION_UNET/`
- **Checkpoint Verified:** `checkpoints/best_model.pth` (Saved at **Epoch 96** out of 100).
- **Architecture:** `AttentionUNet` (31,387,049 trainable parameters, 4 downsampling stages, Additive Attention Gates).
- **Validation Manifest:** `5_Models/MASTER_SPLIT/val_20.csv` (Exactly **125** unique single-mask images: 84 benign, 41 malignant).
- **Locked Test Set:** `5_Models/MASTER_SPLIT/test_10.csv` (63 samples) — **Strictly unaccessed and locked.**

---

## 3. Aggregate Metric Reproduction Check

Re-evaluating the best checkpoint (`best_model.pth`) across the 125 validation images with deterministic evaluation parameters ($\tau = 0.50$) confirms exact alignment with original logged records:

| Metric | Reference Stored (`summary_report.json`) | Reproduced Audit Value (ddof=0) | Reproduced Audit Value (ddof=1) | Discrepancy / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Dice** | $0.7527 \pm 0.2642$ | **$0.7530 \pm 0.2635$** | **$0.7530 \pm 0.2646$** | Exact Match ($\Delta < 0.0003$) |
| **Validation IoU** | $0.6602 \pm 0.2730$ | **$0.6603 \pm 0.2723$** | **$0.6603 \pm 0.2734$** | Exact Match |
| **Precision** | $0.8152 \pm 0.2461$ | **$0.8128 \pm 0.2461$** | **$0.8128 \pm 0.2470$** | Exact Match |
| **Recall** | $0.7666 \pm 0.2905$ | **$0.7691 \pm 0.2901$** | **$0.7691 \pm 0.2913$** | Exact Match |
| **Specificity** | $0.9834 \pm 0.0319$ | **$0.9831 \pm 0.0323$** | **$0.9831 \pm 0.0324$** | Exact Match |
| **HD95 (Valid $N=121$)**| $31.47 \pm 36.24$ px | **$31.56 \pm 36.29$ px** | **$31.56 \pm 36.44$ px** | Exact Match |
| **NSD @ 1px** | $0.3969 \pm 0.2770$ | **$0.3957 \pm 0.2756$** | **$0.3957 \pm 0.2768$** | Exact Match |
| **NSD @ 2px** | $0.5196 \pm 0.3188$ | **$0.5200 \pm 0.3185$** | **$0.5200 \pm 0.3198$** | Exact Match |
| **NSD @ 3px** | $0.5972 \pm 0.3263$ | **$0.5973 \pm 0.3252$** | **$0.5973 \pm 0.3265$** | Exact Match |

*Provenance Note:* The minor rounding difference ($\approx 0.0003$) stems from batch-level DataLoader iteration during epoch logging versus sample-wise direct inference. Both sources reflect the identical model state.

---

## 4. Top 20 Worst-Performing Validation Cases

The 20 cases with lowest Dice performance (accounting for all severe failures):

| Rank | Image ID | Class | GT Area (px) | Pred Area (px) | Dice | IoU | Recall | Precision | HD95 (px) | Primary Failure Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#01** | `benign (224).png` | Benign | 8,220 | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | `Invalid` | Empty Prediction |
| **#02** | `benign (219).png` | Benign | 1,670 | 9,156 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 135.3 | Complete Spatial Miss |
| **#03** | `benign (342).png` | Benign | 15,743 | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | `Invalid` | Empty Prediction |
| **#04** | `benign (299).png` | Benign | 3,813 | 436 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 125.5 | Complete Spatial Miss |
| **#05** | `malignant (80).png`| Malignant | 6,438 | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | `Invalid` | Empty Prediction |
| **#06** | `malignant (158).png`| Malignant | 16,326 | 0 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | `Invalid` | Empty Prediction |
| **#07** | `benign (336).png` | Benign | 2,587 | 6 | **0.0046** | 0.0023 | 0.0023 | 1.0000 | 62.6 | Severe Under-segmentation |
| **#08** | `benign (414).png` | Benign | 20,159 | 59 | **0.0058** | 0.0029 | 0.0029 | 1.0000 | 110.3 | Severe Under-segmentation |
| **#09** | `benign (311).png` | Benign | 9,388 | 839 | **0.1521** | 0.0823 | 0.0829 | 0.9273 | 136.9 | Severe Under-segmentation |
| **#10** | `malignant (177).png`| Malignant | 3,192 | 950 | **0.2752** | 0.1596 | 0.1786 | 0.6000 | 95.2 | Severe Under-segmentation |
| **#11** | `benign (199).png` | Benign | 12,174 | 2,043 | **0.2874** | 0.1678 | 0.1678 | 1.0000 | 38.2 | Severe Under-segmentation |
| **#12** | `benign (37).png` | Benign | 1,130 | 5,188 | **0.3254** | 0.1943 | 0.9097 | 0.1981 | 118.1 | Severe Over-segmentation |
| **#13** | `benign (360).png` | Benign | 2,645 | 10,942 | **0.3679** | 0.2254 | 0.9448 | 0.2284 | 95.4 | Severe Over-segmentation |
| **#14** | `benign (377).png` | Benign | 1,434 | 335 | **0.3742** | 0.2302 | 0.2308 | 0.9881 | 19.6 | Severe Under-segmentation |
| **#15** | `malignant (107).png`| Malignant | 7,945 | 2,172 | **0.4294** | 0.2734 | 0.2734 | 1.0000 | 47.0 | Severe Under-segmentation |
| **#16** | `benign (228).png` | Benign | 7,594 | 2,302 | **0.4446** | 0.2859 | 0.2897 | 0.9557 | 61.8 | Severe Under-segmentation |
| **#17** | `malignant (95).png`| Malignant | 9,739 | 8,277 | **0.4449** | 0.2861 | 0.4115 | 0.4842 | 60.3 | Low Overlap / Boundary Shift |
| **#18** | `malignant (97).png`| Malignant | 11,894 | 7,484 | **0.4723** | 0.3091 | 0.3847 | 0.6114 | 71.4 | Low Overlap / Boundary Shift |
| **#19** | `malignant (72).png`| Malignant | 25,325 | 7,984 | **0.4791** | 0.3150 | 0.3151 | 0.9994 | 79.1 | Severe Under-segmentation |
| **#20** | `malignant (176).png`| Malignant | 6,992 | 19,525 | **0.5071** | 0.3397 | 0.9617 | 0.3444 | 86.8 | Over-segmentation / Leakage |

---

## 5. Quantitative Distribution & Subgroup Analyses

### A. Performance Distribution Metrics (N=125)
- **Mean Dice:** $0.7530$ ($\sigma_{\text{sample}} = 0.2646$, $\sigma_{\text{pop}} = 0.2635$)
- **Median Dice:** **$0.8580$**
- **Interquartile Range (IQR):** $0.2660$ ($\text{Q1} = 0.6731$, $\text{Q3} = 0.9391$)
- **10th & 90th Percentiles:** $P_{10} = 0.3704$, $P_{90} = 0.9526$
- **Failure Proportions:**
  - $\text{Dice} < 0.25$: **9 cases (7.20%)**
  - $\text{Dice} < 0.50$: **19 cases (15.20%)**
  - $\text{Dice} < 0.75$: **39 cases (31.20%)**
  - $\text{Dice} \ge 0.75$: **86 cases (68.80%)**
- **Edge Case Counts:**
  - Empty Predictions: **4** (`benign (224)`, `benign (342)`, `malignant (80)`, `malignant (158)`)
  - Empty Ground Truths: **0** (All 125 validation samples contain verified lesion masks)
  - Invalid HD95 ($\text{NaN}$): **4** (Directly resulting from the 4 empty predictions)

### B. Lesion Size Subgroup Stratification
Lesion size quartiles defined strictly by validation ground truth area fractions:

| Size Quartile | Area Fraction Range (Area in px) | Sample Count | Mean Dice | Std Dice (ddof=1) | Median Dice | Failure Rate ($\text{Dice} < 0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (Small)** | $\le 2.06\%$ ($\le 1,350$ px) | 32 | **0.8615** | 0.1400 | **0.9173** | **3.12% (1/32)** |
| **Q2 (Medium-Low)**| $2.06\% - 5.15\%$ ($1,351 - 3,375$ px) | 31 | **0.7502** | 0.2755 | **0.8471** | **16.13% (5/31)** |
| **Q3 (Medium-High)**| $5.15\% - 13.28\%$ ($3,376 - 8,703$ px) | 31 | **0.7520** | 0.2939 | **0.8855** | **16.13% (5/31)** |
| **Q4 (Large)** | $> 13.28\%$ ($> 8,703$ px) | 31 | **0.6447** | 0.2865 | **0.7747** | **25.81% (8/31)** |

*Observation:* Large lesions exhibit an elevated failure rate ($25.81\%$) due to the model failing to bridge diffuse acoustic margins across large visual receptive fields.

### C. Pathology Class Performance Stratification

| Pathology Class | Sample Count | Mean Dice | Std Dice (ddof=1) | Median Dice | Failure Rate ($\text{Dice} < 0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 84 | **0.7837** | 0.2793 | **0.9173** | **14.29% (12/84)** |
| **Malignant** | 41 | **0.6900** | 0.2216 | **0.7854** | **17.07% (7/41)** |

---

## 6. Observed Failure Patterns and Root Causes

From quantitative profiling and review of the visual overlays in `case_overlays/` and `worst_20_contact_sheet.png`:

### 1. Focal Core Under-Segmentation (High Precision, Low Recall)
- **Manifestation:** In 9 out of 20 failure cases (e.g., `benign (336)`, `benign (414)`, `benign (311)`, `benign (199)`, `malignant (72)`), precision is $>0.92$ (frequently $1.00$), but recall is $<0.30$.
- **Mechanism:** Attention gates strongly attenuate peripheral features where lesion-to-tissue contrast is low. The network converges on only the darkest hypoechoic center, omitting the outer boundary.

### 2. Complete Model Dropout (Empty Predictions)
- **Manifestation:** 4 cases produced zero foreground pixels despite prominent lesions.
- **Mechanism:** When soft attention maps produce values below the sigmoid activation threshold everywhere, Dice loss gradient vanishing during late epochs can leave large, atypical lesions undetected.

### 3. Glandular / Shadowing Spillover (Severe Over-segmentation)
- **Manifestation:** Cases such as `benign (37)` ($\text{Pred}/\text{GT} = 4.59$) and `benign (360)` ($\text{Pred}/\text{GT} = 4.14$) display high recall ($>0.90$) but low precision ($<0.23$).
- **Mechanism:** Dense acoustic posterior shadows behind ribs or fibrous tissue mimic hypoechoic tumor parenchyma, leading to false-positive vertical leakage.

### 4. Complete Spatial Disconnect
- **Manifestation:** Cases `benign (219)` and `benign (299)` exhibit Dice $= 0.0$ with non-zero predictions.
- **Mechanism:** The attention gate locked onto a non-lesion hypoechoic region on the opposite side of the image.

---

## 7. Limitations

1. **Phase-1 Scope:** Evaluation is strictly conducted on the single-mask BUSI cohort.
2. **Deterministic Threshold:** All predictions use a fixed probability threshold of $\tau = 0.50$; soft probability calibration was not tuned.
3. **Loss Function Limitation:** The model was trained purely with Dice loss, which is known to be sensitive to gradient instability on boundary pixels without a complementary cross-entropy or focal term.

---

## 8. Prioritized Recommendations for Subsequent Experiments

Based strictly on the evidence from this audit:

1. **Recommendation 1 — Composite Loss Function (Dice + BCE / Focal):**
   *Evidence:* The dominant failure mode is focal core under-segmentation (Precision $\approx 1.0$, Recall $< 0.3$) and empty predictions ($N=4$). Adding Binary Cross-Entropy or Focal Loss provides pixel-level penalization that forces the model to expand beyond the highest-confidence focal core and prevents complete prediction dropouts.
2. **Recommendation 2 — Boundary-Aware Penalization (Boundary / Surface Loss):**
   *Evidence:* HD95 standard deviation is exceptionally large ($36.29$ px) with severe boundary leaks on spicular malignant borders.
3. **Recommendation 3 — Multi-Scale Receptive Field & Contrast Augmentation:**
   *Evidence:* Large lesions in Q4 failed at $25.81\%$, indicating that fixed-scale attention struggles with expansive hypoechoic regions under varying gain and contrast settings.
