# Final Held-Out Test Evaluation Report: BUSI Attention U-Net

---

## 1. Study Objective
The objective of this investigation is to obtain the final, unbiased held-out generalization performance estimate for the optimal breast ultrasound lesion segmentation configuration identified during prior benchmarking phases. Following benchmark evaluations across network architectures (U-Net, Attention U-Net, Residual U-Net, Attention Residual U-Net, and U-Net++) and loss formulations (Dice baseline, BCE, BCE + Dice, Focal, and Tversky), Attention U-Net trained with combined Binary Cross-Entropy and Soft Dice loss (`BCE_Dice`) was selected based strictly on validation-set performance. This evaluation represents the **first and only** access to the locked, held-out test set for this configuration.

---

## 2. Dataset Information
- **Dataset:** Breast Ultrasound Images (BUSI) dataset (Phase-1 Single-Mask Cohort).
- **Inclusion Criteria:** Single-mask samples from benign and malignant categories; multi-mask anomalies excluded during Phase-1 protocol curation.
- **Split Protocol:** Master Stratified Split (`MASTER_SPLIT`):
  - **Train (`train_70.csv`):** 440 images (70%)
  - **Validation (`val_20.csv`):** 125 images (20%)
  - **Held-Out Test (`test_10.csv`):** 63 images (10%)
- **Data Integrity:** Strict sample-level disjointness between Train, Validation, and Test partitions. Zero overlap verified during pre-evaluation audit.

---

## 3. Test Set Size & Stratification
- **Total Test Samples:** 63 images
- **Class Breakdown:**
  - **Benign:** 42 images (66.67%)
  - **Malignant:** 21 images (33.33%)
- **Test Set Status:** Locked throughout architecture exploration, hyperparameter tuning, loss function benchmarking, and checkpoint selection. Evaluated exactly once.

---

## 4. Frozen Architecture Specification
- **Model:** Attention U-Net (Oktay et al., 2018)
- **Input Channels:** 1 (grayscale ultrasound)
- **Output Channels:** 1 (binary lesion segmentation logits)
- **Base Feature Channels:** 64
- **Decoder Upsampling:** Transposed Convolutions (`bilinear=False`)
- **Attention Gates:** Additive Attention Gates on skip connections

---

## 5. Parameter Count
- **Trainable Parameters:** 31,387,049
- **Non-Trainable Parameters:** 0
- **Total Parameters:** 31,387,049 (verified via exact parameter count audit)

---

## 6. Selected Loss Function
- **Loss Formulation:** Combined Binary Cross-Entropy and Soft Dice Loss (`BCEDiceLoss`)
- **Mathematical Definition:**
  $$\mathcal{L}_{\text{BCE+Dice}} = w_{\text{BCE}} \cdot \mathcal{L}_{\text{BCE}}(\text{logits}, y) + w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}(\sigma(\text{logits}), y)$$

---

## 7. Loss Parameters
- `bce_weight`: 1.0
- `dice_weight`: 1.0
- `smooth`: 1e-6
- `from_logits`: `True`
- `reduction`: `mean`

---

## 8. Selected Checkpoint & Epoch
- **Checkpoint File:** `F:\BUSI-segmentation-architecture-study\5_Models\Best_Architecture\BCE_Dice\checkpoints\best_model.pth`
- **Selected Epoch:** Epoch 97
- **Selection Criterion:** Highest macro-averaged validation Dice score across 100 training epochs on `val_20.csv` (125 samples).

---

## 9. Validation Performance Used for Selection
The checkpoint was selected based strictly on the following validation cohort results:

### Validation Performance (Epoch 97, N=125)
- **Dice Similarity Coefficient:** **0.7572 ± 0.2575**
- **Intersection over Union (IoU):** 0.6650 ± 0.2730
- **Precision:** 0.8188 ± 0.2250
- **Recall (Sensitivity):** 0.7734 ± 0.2830
- **Specificity:** 0.9855 ± 0.0246
- **HD95 (pixels):** 34.16 ± 39.91 px (Valid: 124 / 125 samples; 1 undefined due to empty-mask mismatch)
- **Normalized Surface Distance (NSD) @ 1px:** 0.3957 ± 0.2554
- **Normalized Surface Distance (NSD) @ 2px:** 0.5270 ± 0.3005
- **Normalized Surface Distance (NSD) @ 3px:** 0.6102 ± 0.3103

---

## 10. Final Held-Out Test Set Performance
Evaluated deterministically on the locked test manifest (`test_10.csv`, N=63 samples) using fixed threshold 0.5 without test-time augmentation.

### Overall Test Performance (N=63)
| Metric | Mean | Standard Deviation | Valid Cases / Total | Units / Direction |
| :--- | :---: | :---: | :---: | :--- |
| **Dice Similarity Coefficient (DSC)** | **0.7349** | **0.2749** | 63 / 63 | [0, 1] — Higher is better |
| **Intersection over Union (IoU / Jaccard)** | **0.6401** | **0.2769** | 63 / 63 | [0, 1] — Higher is better |
| **Precision (Positive Predictive Value)** | **0.7641** | **0.2933** | 63 / 63 | [0, 1] — Higher is better |
| **Recall / Sensitivity (True Positive Rate)** | **0.7647** | **0.2771** | 63 / 63 | [0, 1] — Higher is better |
| **Specificity (True Negative Rate)** | **0.9829** | **0.0239** | 63 / 63 | [0, 1] — Higher is better |
| **95th Percentile Hausdorff Distance (HD95)** | **36.75** | **36.58** | 60 / 63 | Pixels — Lower is better |
| **Normalized Surface Distance (NSD @ 1px)** | **0.3213** | **0.2162** | 63 / 63 | [0, 1] — Higher is better |
| **Normalized Surface Distance (NSD @ 2px)** | **0.4458** | **0.2722** | 63 / 63 | [0, 1] — Higher is better |
| **Normalized Surface Distance (NSD @ 3px)** | **0.5398** | **0.2911** | 63 / 63 | [0, 1] — Higher is better |

---

## 11. Per-Class Test Set Breakdown
| Metric | Benign (N=42) | Malignant (N=21) | Overall Cohort (N=63) |
| :--- | :---: | :---: | :---: |
| **Dice (Mean ± Std)** | 0.7748 ± 0.2734 | 0.6551 ± 0.2737 | **0.7349 ± 0.2749** |
| **IoU (Mean ± Std)** | 0.6914 ± 0.2755 | 0.5374 ± 0.2634 | **0.6401 ± 0.2769** |
| **Precision (Mean ± Std)** | 0.7678 ± 0.2985 | 0.7568 ± 0.2894 | **0.7641 ± 0.2933** |
| **Recall (Mean ± Std)** | 0.8210 ± 0.2587 | 0.6519 ± 0.2842 | **0.7647 ± 0.2771** |
| **Specificity (Mean ± Std)** | 0.9851 ± 0.0210 | 0.9784 ± 0.0288 | **0.9829 ± 0.0239** |
| **HD95 (Mean ± Std, px)** | 32.87 ± 36.67 | 44.53 ± 35.11 | **36.75 ± 36.58** |
| **HD95 Valid Count** | 40 / 42 (95.2%) | 20 / 21 (95.2%) | **60 / 63 (95.2%)** |
| **NSD @ 1px (Mean ± Std)** | 0.3906 ± 0.2185 | 0.1828 ± 0.1264 | **0.3213 ± 0.2162** |
| **NSD @ 2px (Mean ± Std)** | 0.5333 ± 0.2662 | 0.2706 ± 0.2010 | **0.4458 ± 0.2722** |
| **NSD @ 3px (Mean ± Std)** | 0.6290 ± 0.2755 | 0.3614 ± 0.2458 | **0.5398 ± 0.2911** |

---

## 12. Comparison: Validation vs. Held-Out Test Set
| Metric | Validation Set (N=125) [Used for Selection] | Held-Out Test Set (N=63) [Unbiased Estimate] | Absolute Difference (Δ) |
| :--- | :---: | :---: | :---: |
| **Dice** | 0.7572 ± 0.2575 | 0.7349 ± 0.2749 | -0.0223 (-2.9%) |
| **IoU** | 0.6650 ± 0.2730 | 0.6401 ± 0.2769 | -0.0249 (-3.7%) |
| **Precision** | 0.8188 ± 0.2250 | 0.7641 ± 0.2933 | -0.0547 (-6.7%) |
| **Recall** | 0.7734 ± 0.2830 | 0.7647 ± 0.2771 | -0.0087 (-1.1%) |
| **Specificity** | 0.9855 ± 0.0246 | 0.9829 ± 0.0239 | -0.0026 (-0.3%) |
| **HD95 (px)** | 34.16 ± 39.91 | 36.75 ± 36.58 | +2.59 px (+7.6%) |
| **HD95 Valid Cases** | 124 / 125 (99.2%) | 60 / 63 (95.2%) | -4.0% |
| **NSD @ 1px** | 0.3957 ± 0.2554 | 0.3213 ± 0.2162 | -0.0744 (-18.8%) |
| **NSD @ 2px** | 0.5270 ± 0.3005 | 0.4458 ± 0.2722 | -0.0812 (-15.4%) |
| **NSD @ 3px** | 0.6102 ± 0.3103 | 0.5398 ± 0.2911 | -0.0704 (-11.5%) |

### Observations:
1. **Generalization Gap:** The test Dice score of **0.7349** reflects an expected, modest drop of 0.0223 (-2.9%) relative to the validation Dice (**0.7572**), demonstrating solid generalization without catastrophic overfitting.
2. **Class Disparity:** The model demonstrates stronger segmentation performance on benign lesions (Dice: 0.7748 ± 0.2734) than on malignant lesions (Dice: 0.6551 ± 0.2737). Malignant lesions feature irregular, infiltrative boundaries and acoustic shadowing, posing greater segmentation challenge.
3. **Boundary Fidelity:** HD95 increases modestly from 34.16 px to 36.75 px. The valid count on the test set is 60/63 due to 3 cases where the model predicted zero positive pixels on low-contrast lesions.

---

## 13. Per-Image Metric Details & NaN / Invalid Handling
- **Full Per-Image Metrics File:** `test_per_image_metrics.csv` (contains all 63 rows).
- **Valid Counts:**
  - Dice, IoU, Precision, Recall, Specificity, NSD @ 1,2,3 px: **63 / 63 valid** (100%).
  - HD95: **60 / 63 valid** (95.2%).
- **Cases with NaN HD95 (Zero Positive Prediction):**
  1. `benign (212).png`: Ground Truth = 5,588 px, Prediction = 0 px (Dice = 0.0000, HD95 = NaN)
  2. `benign (416).png`: Ground Truth = 1,963 px, Prediction = 0 px (Dice = 0.0000, HD95 = NaN)
  3. `malignant (81).png`: Ground Truth = 5,812 px, Prediction = 0 px (Dice = 0.0000, HD95 = NaN)
- **Mathematical Justification:** Under standard surface distance formulations, the Hausdorff distance between an empty predicted boundary set and a non-empty ground-truth contour is mathematically undefined. Consistent with project rules, these values are preserved as `NaN` rather than artificially set to zero.

---

## 14. Exact Preprocessing & Inference Pipeline
- **Input Channels:** 1 channel (grayscale)
- **Spatial Resizing:** 256 × 256 pixels
  - Image interpolation: Bilinear (`PIL.Image.Resampling.BILINEAR`)
  - Mask interpolation: Nearest Neighbor (`PIL.Image.Resampling.NEAREST`)
- **Intensity Normalization:** Scaled strictly to $[0.0, 1.0]$ via standard tensor conversion (`TF.to_tensor`)
- **Test-Time Augmentation (TTA):** None (strictly deterministic inference)
- **Inference Threshold:** Fixed at **0.5** applied to sigmoid probabilities:
  $$\hat{Y} = (\sigma(\text{logits}) > 0.5)$$
- **Batch Size:** 1 (isolated per-sample evaluation)

---

## 15. Artifacts and File Locations
All evaluation artifacts are preserved in `5_Models\Best_Architecture\FINAL_TEST_EVALUATION\`:
- **Checkpoint Metadata:** [`checkpoint_info.json`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/checkpoint_info.json)
- **Aggregate Test Metrics:** [`test_aggregate_metrics.json`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/test_aggregate_metrics.json)
- **Per-Sample Metrics CSV:** [`test_per_image_metrics.csv`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/test_per_image_metrics.csv)
- **Predicted Binary Masks (63 PNGs):** `FINAL_TEST_EVALUATION/test_predictions/`
- **Qualitative Visualizations:**
  - Prediction Grid: [`visualizations/test_qualitative_predictions.png`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/visualizations/test_qualitative_predictions.png)
  - Metric Distributions: [`visualizations/test_metric_distributions.png`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/visualizations/test_metric_distributions.png)
- **Execution Log:** [`logs/test_eval.log`](file:///f:/BUSI-segmentation-architecture-study/5_Models/Best_Architecture/FINAL_TEST_EVALUATION/logs/test_eval.log)

---

## 16. Research Integrity & Model Selection Statement
> **Integrity Confirmation:**
> The test set (`test_10.csv`, N=63) was completely locked and isolated throughout all experimental stages. It was **NOT** used for model selection, hyperparameter tuning, loss function selection, architecture selection, checkpoint selection, or threshold tuning.
>
> Combined Binary Cross-Entropy and Soft Dice loss (`BCE_Dice`) was selected based solely on validation-set performance (`val_20.csv`, N=125) and subsequently evaluated exactly once on the locked held-out test set to produce an unbiased estimate of generalization performance.

---

## 17. Limitations and Caveats
1. **Sample Size:** The held-out test set comprises 63 images (42 benign, 21 malignant). While adhering strictly to the predefined 70/20/10 split protocol, the sample size reflects standard dataset constraints in medical imaging research.
2. **False Negatives:** In 3 out of 63 test images (4.8%), the model generated zero positive foreground predictions. These failure cases correspond to subtle lesions with low acoustic contrast or severe speckle artifacts.
3. **Class Performance Difference:** Malignant lesions exhibited noticeably lower average Dice (0.6551) compared to benign lesions (0.7748), reflecting greater morphological complexity and ill-defined margins.
4. **Statistical Framing:** BCE + Dice is **not** claimed as "proven superior" to all alternatives on the test set; no comparative models were run on the test set, in strict compliance with single-evaluation protocol rules.
