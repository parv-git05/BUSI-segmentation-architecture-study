# 01_UNet: Standard U-Net Architecture Implementation

**Workstream Location:** `F:\Research_BUSI_Segmentation\5_Models\01_UNet\`  
**Target Model:** 2D Encoder-Decoder U-Net (Ronneberger et al., 2015)  
**Status:** `IMPLEMENTED & INTEGRATION TESTED` (22 / 22 PyTest Unit Tests Passing)

---

## 1. Architectural Specification

The U-Net architecture follows the classic symmetric encoder-decoder topology with skip connections:

- **Input Tensor Shape:** $[B, 1, 256, 256]$ (Grayscale ultrasound images)
- **Output Tensor Shape:** $[B, 1, 256, 256]$ (Raw unnormalized lesion segmentation logits)
- **Encoder Path (Contracting):**
  - Level 0 (In Conv): $64$ channels
  - Level 1 (Down 1): $128$ channels
  - Level 2 (Down 2): $256$ channels
  - Level 3 (Down 3): $512$ channels
  - Level 4 (Bottleneck): $1024$ channels
- **Decoder Path (Expanding):**
  - Level 3 (Up 1): $1024 \to 512$ channels (Concat skip from Encoder Level 3)
  - Level 2 (Up 2): $512 \to 256$ channels (Concat skip from Encoder Level 2)
  - Level 1 (Up 3): $256 \to 128$ channels (Concat skip from Encoder Level 1)
  - Level 0 (Up 4): $128 \to 64$ channels (Concat skip from Encoder Level 0)
- **Convolution Structure:** Double $3 \times 3$ Convolution $\to$ 2D Batch Normalization $\to$ ReLU ($\text{padding} = 1$).
- **Downsampling Method:** Max Pooling $2 \times 2$ with $\text{stride} = 2$.
- **Upsampling Method:** Transposed 2D Convolution ($2 \times 2$, $\text{stride} = 2$).
- **Final Output Layer:** $1 \times 1$ Convolution ($64 \to 1$ channel).
- **Trainable Parameter Count:** **31,036,481 parameters** (31.04M).

---

## 2. Inherited Baseline Research Protocol

All settings are strictly inherited from `00_Common`:

- **Dataset:** BUSI Phase-1 (628 samples: 420 benign, 208 malignant)
- **Split:** Train = 440, Validation = 125, Test = 63 (Stratified, `Seed = 42`)
- **Input Size:** $256 \times 256$, Grayscale (1 channel)
- **Train Augmentation:** Horizontal flip ($p=0.5$), Vertical flip ($p=0.5$), Rotation $\pm 15^\circ$
- **Val / Test:** Preprocessing only (Resize + Normalization), No random augmentation, No TTA
- **Loss Function:** Binary Dice Loss ($\text{smooth} = 10^{-6}$)
- **Optimizer:** Adam ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau` ($\text{patience} = 10$, $\text{factor} = 0.5$)
- **Epoch Budget:** 100 Epochs (Early Stopping **DISABLED**)
- **Best Checkpoint Criteria:** Maximum Validation Dice Score
- **Locked Test Set Policy:** Test set (`test_10.csv`, 63 samples) is **LOCKED** and untouched.

---

## 3. Integration with `00_Common`

The U-Net model connects directly to the shared research components without duplicating common logic:
- `BUSIDataset` & `create_dataloader` from `00_Common.datasets`
- `get_train_transforms` & `get_val_transforms` from `00_Common.datasets`
- `DiceLoss` from `00_Common.training`
- `SegmentationTrainer` from `00_Common.training`
- `save_checkpoint` & `load_checkpoint` from `00_Common.utils`
- `ExperimentTracker` from `00_Common.utils`
- `calculate_batch_metrics` & `aggregate_metrics` from `00_Common.metrics`

---

## 4. Pipeline Smoke Test Summary

> [!NOTE]
> **DISCLAIMER: PIPELINE INTEGRATION TEST ONLY**  
> The 2-epoch smoke test was executed exclusively to verify end-to-end integration across CUDA hardware, data loading, gradient backpropagation, loss computation, metric aggregation, checkpointing, and visualization.  
> **These smoke test metrics are NOT research results**, must NOT be placed in the final architecture comparison table, and must NOT be used for model selection.

- **Smoke Run Execution Date:** 2026-09-29
- **Hardware:** CUDA GPU (`Device: cuda`)
- **Duration:** 41.8 seconds (2 epochs, batch size 8)
- **Output Storage:** `01_UNet/smoke_test/`
  - `config.json`
  - `train.log`
  - `checkpoints/` (`best_model.pth`, `latest.pth`)
  - `training_curves/`
  - `qualitative_predictions/`
  - `summary_report.json`

---

## 5. Official 100-Epoch Experiment Preparation

The official experiment directory structure has been prepared at:
`01_UNet/experiments/EXP-ARCH-01_UNET/`

- **Status:** **PREPARED BUT NOT EXECUTED.**
- **Config:** `EXP-ARCH-01_UNET/config.json` generated.
- No fake results or placeholders were created.
