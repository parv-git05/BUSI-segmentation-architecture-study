# BUSI Phase-1 Master Split Verification Report

**Status:** `VERIFIED`  
**Date:** 2026-09-29 12:14:22  
**Dataset Location:** `F:\Research_BUSI_Segmentation\`  
**Master Split Directory:** `F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT\`  
**Random Seed:** `42`

---

## 1. Executive Summary

The BUSI Phase-1 dataset master split has been deterministically generated and programmatically audited. All 628 Phase-1 eligible single-mask ultrasound samples (420 benign, 208 malignant) were assigned to strictly disjoint train (70%), validation (20%), and test (10%) splits using stratified sampling.

Stale absolute paths beginning with `D:\Dataset_BUSI_with_GT\` were successfully rebased to `F:\Research_BUSI_Segmentation\`, and 100% of all image and mask paths were verified to exist on local disk.

---

## 2. Split Composition & Class Distribution

| Split | Total Samples | % of Cohort | Benign (Class 0) | Malignant (Class 1) | Benign % | Malignant % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (70%)** | 440 | 70.06% | 294 | 146 | 66.82% | 33.18% |
| **Val (20%)** | 125 | 19.90% | 84 | 41 | 67.20% | 32.80% |
| **Test (10%)** | 63 | 10.03% | 42 | 21 | 66.67% | 33.33% |
| **Total Cohort** | **628** | **100.00%** | **420** | **208** | **66.88%** | **33.12%** |

*Note: Due to integer rounding over 628 samples, the exact split percentages are 70.06% Train (440), 19.90% Val (125), and 10.03% Test (63).*

---

## 3. Programmatic Verification & Audit Matrix

| Audit Check | Requirement | Result | Status |
| :--- | :--- | :---: | :---: |
| **Train ∩ Val Overlap** | Must be 0 samples | `0` | **PASS** |
| **Train ∩ Test Overlap** | Must be 0 samples | `0` | **PASS** |
| **Val ∩ Test Overlap** | Must be 0 samples | `0` | **PASS** |
| **Total Unique Samples** | Must equal 628 | `628` | **PASS** |
| **Split Count Sum** | Train + Val + Test = 628 | `628` | **PASS** |
| **No Duplicate Images** | All image paths unique | `628` / `628` | **PASS** |
| **No Duplicate Masks** | All mask paths unique | `628` / `628` | **PASS** |
| **Image Paths Exist** | All image files on disk | 0 missing | **PASS** |
| **Mask Paths Exist** | All mask files on disk | 0 missing | **PASS** |
| **Pair Correspondence** | Image stem matches mask stem | 0 mismatches | **PASS** |

---

## 4. Path Rebasing Verification

- **Original Path Prefix:** `D:\Dataset_BUSI_with_GT\`
- **Rebased Path Prefix:** `F:\Research_BUSI_Segmentation\`
- **Original Manifest Files Unmodified:**
  - `BUSI_phase1_manifest.csv` (Preserved intact)
  - `BUSI_phase1_excluded.csv` (Preserved intact)
- **Disk Existence Check:** 1,256 total files (628 images + 628 masks) verified present on storage.

---

## 5. Mandatory Research Policy Notice

> [!IMPORTANT]
> **FINAL LOCKED TEST SET POLICY**  
> The test set (`test_10.csv`, 63 samples) is the **FINAL LOCKED TEST SET**.  
> Under no circumstances should the test set be used for:
> - Model architecture selection or hyperparameter tuning
> - Loss function tuning or epoch/early-stopping criteria selection
> - Iterative validation or model checkpointing  
> Test evaluation must occur exactly once at the final benchmark phase.

---

## 6. Generated Manifest Artifacts

All files are written under `F:\Research_BUSI_Segmentation\5_Models\MASTER_SPLIT\`:

1. `master_manifest.csv` (628 rows)
2. `train_70.csv` (440 rows)
3. `val_20.csv` (125 rows)
4. `test_10.csv` (63 rows)
5. `split_metadata.json` (Structured metadata & audit log)
6. `SPLIT_VERIFICATION_REPORT.md` (This report)

---
*Report generated automatically by Antigravity AI Assistant.*
