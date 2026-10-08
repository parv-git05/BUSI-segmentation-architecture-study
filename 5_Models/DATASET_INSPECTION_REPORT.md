# BUSI Phase-1 Dataset Inspection Report

**Date:** September 28, 2026  
**Workstream:** `F:\Research_BUSI_Segmentation\5_Models\`  
**Target Study:** BUSI Breast Ultrasound Lesion Segmentation Study (U-Net, Attention U-Net, Residual U-Net, Attention Residual U-Net, U-Net++)

---

## 1. Executive Summary & Verification

| Parameter | Expected Phase-1 Cohort | Actual Verified Cohort | Status |
| :--- | :--- | :--- | :--- |
| **Total Eligible Samples** | 628 | 628 | Verified |
| **Benign Class Count** | 420 | 420 | Verified |
| **Malignant Class Count** | 208 | 208 | Verified |
| **Normal Class Count** | 0 (Excluded) | 0 (In Manifest) | Verified |
| **Single-Mask Verification** | 628 | 628 (100% 1:1) | Verified |
| **Corrupt / Unreadable Files**| 0 | 0 | Verified |
| **Duplicate Images / Masks** | 0 | 0 | Verified |
| **Dimension Mismatches** | 0 | 0 (100% match) | Verified |

---

## 2. Dataset Source & Location

- **Root Directory:** `F:\Research_BUSI_Segmentation\`
- **Authoritative Manifest:** `F:\Research_BUSI_Segmentation\BUSI_phase1_manifest.csv`
- **Exclusion Manifest:** `F:\Research_BUSI_Segmentation\BUSI_phase1_excluded.csv`
- **Data Subdirectories:**
  - `F:\Research_BUSI_Segmentation\benign\` (436 image PNGs, 453 mask PNGs)
  - `F:\Research_BUSI_Segmentation\malignant\` (209 image PNGs, 210 mask PNGs)
  - `F:\Research_BUSI_Segmentation\normal\` (133 image PNGs, 133 mask PNGs — excluded from Phase-1)
  - `F:\Research_BUSI_Segmentation\problems in multi masks\` (9 files: qualitative visual note PNGs & PDF)

---

## 3. Manifest Schemas & Structure

### A. Phase-1 Manifest (`BUSI_phase1_manifest.csv`)
- **Total Rows:** 628 (excluding header)
- **Columns (6):**
  1. `image_path` (string): Absolute file path to ultrasound image
  2. `mask_path` (string): Absolute file path to single ground-truth lesion mask
  3. `class` (string): Lesion class (`benign` or `malignant`)
  4. `class_id` (integer): Binary class index (`0` for benign, `1` for malignant)
  5. `num_masks` (integer): Number of ground-truth masks attached (`1` for all 628 Phase-1 samples)
  6. `use_phase1` (integer): Flag indicating Phase-1 inclusion (`1` for all 628 samples)

### B. Phase-1 Excluded (`BUSI_phase1_excluded.csv`)
- **Total Rows:** 17 (excluding header)
- **Columns (3):**
  1. `image` (string): Absolute file path to ultrasound image
  2. `reason` (string): Cause of exclusion (`multiple_masks` for all 17 samples)
  3. `num_masks` (integer): Number of separate ground-truth mask files (`2` or `3`)

---

## 4. File Path Discrepancy (Crucial Finding)

> [!WARNING]
> **Drive Letter / Path Prefix Discrepancy**
>
> The manifest CSV files store absolute paths formatted as `D:\Dataset_BUSI_with_GT\...` (e.g., `D:\Dataset_BUSI_with_GT\benign\benign (1).png`).
> On the current working filesystem, all files reside under `F:\Research_BUSI_Segmentation\...`.
>
> **Verification Result:**
> - Files existing on `D:\`: **0 / 628**
> - Files existing on `F:\`: **628 / 628** (100% present on disk)
>
> **Action Required for Data Loaders:** Data loaders must resolve relative paths or dynamically rebase paths from `D:\Dataset_BUSI_with_GT\` to `F:\Research_BUSI_Segmentation\`.

---

## 5. Disk Directory & File Reconciliation

All 645 benign and malignant images present on disk are 100% accounted for:

$$\text{Total Benign Images on Disk (436)} = 420 \text{ (Phase-1 Manifest)} + 16 \text{ (Phase-1 Excluded)}$$

$$\text{Total Malignant Images on Disk (209)} = 208 \text{ (Phase-1 Manifest)} + 1 \text{ (Phase-1 Excluded)}$$

$$\text{Total Malignant + Benign Images on Disk} = 645 \quad (\text{Manifest: } 628 + \text{Excluded: } 17 = 645)$$

- **Unaccounted Images / Masks on Disk:** 0
- **Missing / Broken Files:** 0 (All 628 image files and 628 mask files read cleanly without PIL errors)

---

## 6. Image & Mask Integrity Analysis

- **Spatial Resolution & Dimensions:** 100% of image-mask pairs have identical height and width dimensions (no shape mismatch).
- **Mask Binary Pixel Values:** All mask files contain strictly binary mask values (`0` for background, `255` for lesion / `[False, True]`).
- **Corrupt File Check:** 0 corrupt or truncated PNG files detected.

---

## 7. Duplicate & Multi-Mask Findings

- **Duplicate Paths:** 0 duplicate `image_path`, 0 duplicate `mask_path`.
- **Duplicate Identifiers:** 0 duplicate sample IDs (e.g., `benign (1)`, `malignant (10)`).
- **Duplicate Content (MD5 Hashes):** 0 identical duplicate images detected across the cohort.
- **Multi-Mask Findings:**
  - In `BUSI_phase1_manifest.csv`: 0 samples have multiple masks (`num_masks = 1` for all 628).
  - 17 multi-mask images (16 benign, 1 malignant) are completely excluded from Phase-1 cohort and cataloged in `BUSI_phase1_excluded.csv`.

---

## 8. Potential Leakage Concerns & Mitigation

1. **Patient-Level Identification:**
   - BUSI filenames do not contain explicit patient ID metadata (files are named sequentially like `benign (1).png`, `benign (2).png`).
   - MD5 hash analysis confirmed zero identical duplicate images within the Phase-1 cohort.
2. **Stratified Splitting Strategy:**
   - Because Phase-1 consists of 628 single-mask independent images without patient grouping tags, a deterministic 70% train / 20% validation / 10% test split stratified by `class_id` (`0` benign vs `1` malignant) is methodologically sound and leak-free.

---

## 9. Conclusion

### **READY FOR SPLIT**

The Phase-1 cohort of **628 single-mask images** (420 benign, 208 malignant) is fully verified, intact, uncorrupted, and 100% reconciled against the filesystem.

*Note for subsequent workstream steps:* When creating data loaders or splitting utilities, ensure absolute paths starting with `D:\Dataset_BUSI_with_GT\` are converted to point to `F:\Research_BUSI_Segmentation\`.
