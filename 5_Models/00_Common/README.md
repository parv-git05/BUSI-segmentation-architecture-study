# BUSI Phase-1 Common Research Infrastructure (`00_Common`)

**Workstream Location:** `F:\Research_BUSI_Segmentation\5_Models\00_Common\`  
**Target Benchmarks:** U-Net, Attention U-Net, Residual U-Net, Attention Residual U-Net, U-Net++  
**Status:** `VERIFIED & FULLY TESTED` (16 / 16 PyTest Unit Tests Passing)

---

## 1. Overview & Purpose

This directory contains the unified, reusable, architecture-agnostic research and training infrastructure for the BUSI Phase-1 Breast Ultrasound Lesion Segmentation Study.

To ensure strict scientific control, all five segmentation architectures will execute using:
- **Same dataset & fixed split** (`train_70.csv`, `val_20.csv`, `test_10.csv`)
- **Same preprocessing & binary mask encoding**
- **Same metric implementations** (Dice, IoU, Precision, Recall, Specificity, HD95, NSD @ 1,2,3 px)
- **Same Dice Loss implementation**
- **Same optimization & learning rate protocol**
- **Same training engine, checkpointing, and reproducibility seed policy (`seed = 42`)**

---

## 2. Directory Structure

```
00_Common/
├── README.md                          # Detailed infrastructure documentation (This file)
├── configs/
│   ├── __init__.py
│   └── experiment_config.py           # Dataclass for centralized experiment parameters
├── datasets/
│   ├── __init__.py
│   ├── busi_dataset.py                # Dataset & DataLoader reading strictly from MASTER_SPLIT CSVs
│   └── transforms.py                  # Preprocessing & augmentation pipelines (Train vs Val/Test)
├── metrics/
│   ├── __init__.py
│   └── segmentation_metrics.py        # Standardized Dice, IoU, Precision, Recall, Specificity, HD95, NSD @ 1,2,3px
├── training/
│   ├── __init__.py
│   ├── losses.py                      # Isolated Dice Loss implementation
│   └── trainer.py                     # Architecture-agnostic PyTorch Segmentation Trainer
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py                  # Atomic checkpoint save/load utility (.pth)
│   ├── experiment_tracker.py          # Standardized directory & artifact tracker
│   ├── logger.py                      # Logging configuration
│   └── seed.py                        # Reproducibility RNG seeding (Python, NumPy, PyTorch, CUDA)
├── visualization/
│   ├── __init__.py
│   └── visualizer.py                  # Loss curves, qualitative grids, and metric distribution plots
└── tests/                             # Full PyTest Unit Test Suite
    ├── __init__.py
    ├── test_checkpointing.py
    ├── test_dataset.py
    ├── test_loss.py
    ├── test_metrics.py
    ├── test_preprocessing.py
    └── test_trainer.py
```

---

## 3. Dataset & Split Policy

- **Source Manifests:**
  - `MASTER_SPLIT/master_manifest.csv` (628 eligible single-mask samples)
  - `MASTER_SPLIT/train_70.csv` (440 samples: 294 benign, 146 malignant)
  - `MASTER_SPLIT/val_20.csv` (125 samples: 84 benign, 41 malignant)
  - `MASTER_SPLIT/test_10.csv` (63 samples: 42 benign, 21 malignant — **FINAL LOCKED TEST SET**)
- **Path Resolution:** Dynamically resolves rebased paths (`F:\Research_BUSI_Segmentation\` or `D:\Dataset_BUSI_with_GT\`) to local storage.
- **Grayscale Handling:** Loads images as 1-channel grayscale tensors normalized to $[0.0, 1.0]$.
- **Binary Mask Encoding:** Thresholds ground-truth lesion masks strictly to $\{0.0, 1.0\}$ float32 tensors.

---

## 4. Preprocessing & Augmentation Strategy

- **Spatial Resizing:** Image resized via `BILINEAR` interpolation; Mask resized via `NEAREST` neighbor to avoid continuous boundary blurring.
- **Train Split:** Standard preprocessing + optional geometric augmentation (random horizontal/vertical flips, subtle rotation).
- **Validation Split:** Preprocessing only (Resize + Normalization). **No random augmentations.**
- **Test Split:** Preprocessing only (Resize + Normalization). **No random augmentations. No Test-Time Augmentation (TTA).**

---

## 5. Metric Suite & Definitions

Every model architecture is evaluated against identical metric logic:

1. **Dice Similarity Coefficient (DSC):**
   $$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
2. **Intersection over Union (IoU / Jaccard):**
   $$\text{IoU} = \frac{TP}{TP + FP + FN}$$
3. **Pixel Precision:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
4. **Pixel Recall (Sensitivity):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
5. **Pixel Specificity:**
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
6. **95th Percentile Hausdorff Distance (HD95):**
   95th percentile distance between surface contour points $S_P$ and $S_G$.
   - **Edge Case (Both Empty):** Returns $0.0$ (perfect match of empty sets).
   - **Edge Case (One Empty):** Returns `NaN` (logged and excluded from distance average).
7. **Normalized Surface Distance (NSD @ 1px, 2px, 3px):**
   $$\text{NSD}(\tau) = \frac{|\{ p \in S_P : d(p, S_G) \le \tau \}| + |\{ q \in S_G : d(q, S_P) \le \tau \}|}{|S_P| + |S_G|}$$
   - **Edge Case (Both Empty):** Returns $1.0$.
   - **Edge Case (One Empty):** Returns $0.0$.

---

## 6. Loss Function

- **Dice Loss:**
  $$\text{Loss}_{\text{Dice}} = 1.0 - \frac{2 \sum (P \cdot G) + \text{smooth}}{\sum P + \sum G + \text{smooth}}$$
- Isolated in `training/losses.py` for seamless future extensions.

---

## 7. Standardized Experiment Output Schema

Future architecture runs will automatically format results under:
```
experiments/<exp_name>/
├── config.json
├── train.log
├── checkpoints/
│   ├── best_model.pth
│   └── latest.pth
├── training_curves/
│   ├── epoch_history.csv
│   └── learning_curves.png
├── per_image_metrics/
│   └── best_val_per_image_metrics.csv
├── qualitative_predictions/
│   └── epoch_XXX_qualitative.png
└── summary_report.json
```

---

## 8. Unit Test Execution Results

Executed command:
`pytest F:\Research_BUSI_Segmentation\5_Models\00_Common\tests -v`

Summary: **16 passed in 5.49s**

| Test Module | Test Case | Status |
| :--- | :--- | :---: |
| `test_dataset.py` | `test_manifest_existence_and_counts` | **PASS** |
| `test_dataset.py` | `test_no_split_overlap` | **PASS** |
| `test_dataset.py` | `test_dataset_loading_and_tensors` | **PASS** |
| `test_dataset.py` | `test_dataloader_creation` | **PASS** |
| `test_preprocessing.py` | `test_val_test_deterministic_preprocessing` | **PASS** |
| `test_preprocessing.py` | `test_mask_binary_discrete_preservation` | **PASS** |
| `test_metrics.py` | `test_perfect_match_metrics` | **PASS** |
| `test_metrics.py` | `test_partial_overlap_metrics` | **PASS** |
| `test_metrics.py` | `test_both_empty_edge_case` | **PASS** |
| `test_metrics.py` | `test_one_empty_edge_case` | **PASS** |
| `test_metrics.py` | `test_aggregation` | **PASS** |
| `test_loss.py` | `test_dice_loss_perfect_prediction` | **PASS** |
| `test_loss.py` | `test_dice_loss_from_logits` | **PASS** |
| `test_checkpointing.py` | `test_save_and_load_checkpoint` | **PASS** |
| `test_trainer.py` | `test_reproducibility_seed` | **PASS** |
| `test_trainer.py` | `test_trainer_smoke_execution` | **PASS** |

---

## 9. Unresolved Methodological Decisions Requiring Scientific Approval

Before initiating full model training for the five architectures, the following scientific/methodological decisions require explicit approval:

1. **Target Image Input Size:**  
   - Infrastructure default set to `(256, 256)`.  
   - *Options for approval:* `(256, 256)` vs `(512, 512)` vs native size interpolation.
2. **Training Data Augmentation Policy:**  
   - Infrastructure supports geometric augmentations (random horizontal/vertical flips, rotation $\pm 15^\circ$).  
   - *Options for approval:* No augmentation vs geometric flips/rotation vs intensity scaling.
3. **Training Budget & Early Stopping Epoch Count:**  
   - Default set to `100` epochs with `ReduceLROnPlateau` scheduler.  
   - *Options for approval:* Fixed 100 epochs vs 150 epochs with patience = 15.
4. **Target Checkpoint Selection Metric:**  
   - Default set to `val_dice`.  
   - *Options for approval:* `val_dice` vs `val_loss` vs `val_iou`.
