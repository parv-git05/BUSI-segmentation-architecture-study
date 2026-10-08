# BUSI Attention U-Net Validation Failure Audit & Reproducibility Protocol

## 1. Audit Specification & Provenance

- **Study / Experiment Name:** `EXP-ARCH-02_ATTENTION_UNET`
- **Architecture:** Attention U-Net (Additive Attention Gates, Oktay et al., 2018)
- **Model Checkpoint Path:** `5_Models/02_Attention_UNet/experiments/EXP-ARCH-02_ATTENTION_UNET/checkpoints/best_model.pth`
- **Checkpoint Epoch:** Epoch 96 (out of 100 trained)
- **Manifest Path:** `5_Models/MASTER_SPLIT/val_20.csv`
- **Manifest Sample Count:** 125 validation images (84 Benign, 41 Malignant)
- **Locked Test Set Status:** `5_Models/MASTER_SPLIT/test_10.csv` (63 samples) — **LOCKED & UNTOUCHED (0 test evaluations performed)**
- **Audit Directory:** `5_Models/02_Attention_UNet/analysis/per_image_validation_audit/`

---

## 2. Preprocessing & Inference Protocol

- **Input Dimensions:** $256 \times 256$ pixels (Single-channel Grayscale).
- **Image Interpolation:** `PIL.Image.Resampling.BILINEAR`.
- **Mask Interpolation:** `PIL.Image.Resampling.NEAREST` (preserving discrete binary values).
- **Normalization:** Range $[0.0, 1.0]$.
- **Inference Mode:** `torch.no_grad()`, `model.eval()`, deterministic validation preprocessing without test-time augmentations.
- **Logit Activation:** `torch.sigmoid(logits)`.
- **Binarization Threshold:** $\tau = 0.50$ (`preds_binary = (preds_prob > 0.5).to(torch.float32)`).

---

## 3. Mathematical Metric Definitions

1. **Pixel Confusion Elements:**
   $$\text{TP} = |P \cap G|,\quad \text{FP} = |P \setminus G|,\quad \text{FN} = |G \setminus P|,\quad \text{TN} = |(P \cup G)^c|$$
2. **Dice Similarity Coefficient (DSC):**
   $$\text{DSC} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
   *Edge Case:* If $|P| = 0$ and $|G| = 0$, $\text{DSC} = 1.0$.
3. **Intersection over Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
   *Edge Case:* If $|P| = 0$ and $|G| = 0$, $\text{IoU} = 1.0$.
4. **Precision (Positive Predictive Value):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   *Edge Case:* If $\text{TP} + \text{FP} = 0$, $\text{Precision} = 1.0$ if $|G| = 0$ else $0.0$.
5. **Recall / Sensitivity:**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *Edge Case:* If $\text{TP} + \text{FN} = 0$, $\text{Recall} = 1.0$ if $|P| = 0$ else $0.0$.
6. **Specificity (True Negative Rate):**
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
7. **95th Percentile Hausdorff Distance (HD95):**
   Computed on boundary contours $S_p = \partial P$ and $S_g = \partial G$ via Euclidean Distance Transform ($\text{EDT}$).
   $$\text{HD95} = \text{percentile}_{95}\left( \{d(p, S_g) \mid p \in S_p\} \cup \{d(q, S_p) \mid q \in S_g\} \right)$$
   *Edge Case:* If both empty, $\text{HD95} = 0.0$. If one is empty and the other is non-empty, $\text{HD95} = \text{NaN}$ (recorded as `invalid`).
8. **Normalized Surface Distance (NSD @ $\tau$):**
   $$\text{NSD}(\tau) = \frac{|\{p \in S_p \mid d(p, S_g) \le \tau\}| + |\{q \in S_g \mid d(q, S_p) \le \tau\}|}{|S_p| + |S_g|}$$

---

## 4. Standard Deviation & Aggregation Convention

- Population standard deviation (`ddof=0` / `np.std` / `np.nanstd`) is the standardized repository convention used in `summary_report.json` and official logs.
- Sample standard deviation (`ddof=1` / `pandas.Series.std`) is explicitly reported alongside population statistics in the audit report for complete statistical transparency.

---

## 5. Execution Commands to Reproduce Audit

```bash
# Activate environment
.venv\Scripts\activate

# 1. Run visual panels and contact sheet generation
python 5_Models/02_Attention_UNet/analysis/generate_visual_panels.py

# 2. Run quantitative distribution, lesion size, and class analysis
python 5_Models/02_Attention_UNet/analysis/generate_plots_and_stats.py
```
