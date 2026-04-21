# Recall Improvement Plan — Adipose & Fascicle in Bio-Aegis

Date: 2026-04-21  
Diagnosis by: code audit + quantitative class-distribution analysis

---

## Root Cause Summary

The model under-predicts adipose (~4× below GT) and mildly under-predicts fascicle
in Bio-Aegis images. Three concrete causes were identified:

| # | Issue | Evidence |
|---|---|---|
| 1 | Wrong CE weights | Weights were estimated from a 300-sample dominated 161:1 by GTEx tiles; adipose was measured at 19.93% (GTEx) but is only 6.4% in the actual tile distribution → weight was 0.77 (below average) instead of ~1.74 (above average) |
| 2 | Uniform α,β across all Tversky classes | FN penalty β=0.7 is the same for background (57% of pixels) and adipose (6%); adipose/fascicle never get a recall-biased gradient distinct from easy classes |
| 3 | Foreground oversampling not targeted | `oversample_foreground_percent=0.33` samples any foreground pixel; ~99% of foreground tiles are GTEx connective/fascicle, so adipose patches are almost never oversampled |

---

## Actual Class Frequencies (measured on 500 random training tiles)

```
Class 0  Background              57.51%  → CE weight: 0.1925
Class 1  Connective/Perineurium  23.32%  → CE weight: 0.4746
Class 2  Adipose                  6.37%  → CE weight: 1.7389  ← was 0.772 (WRONG)
Class 3  NerveFascicle           12.80%  → CE weight: 0.8647  ← was 1.082 (ok)
Class 4  Blood_vessel             0.00%  → CE weight: 1.7294  (capped at 2×cls3)
```

The old weights were derived from a GTEx-dominated sample and misrepresented adipose
as a majority class. The new weights correctly reflect training tile reality.

---

## Improvement Ideas (prioritised)

### ✅ Implemented in this branch

#### A. Fix CE class weights (both trainers)
**File:** `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdamEarlyStopping.py`  
**Change:** Update `ce_weights` tensor from old GTEx-estimate to tile-measured values.  
**Impact:** Immediately fixes the "adipose treated as easy class" problem in CE gradient.

#### B. Per-class α,β Tversky + Focal Tversky
**File:** `nnunetv2/training/loss/tversky.py`  
**New class:** `PerClassFocalTverskyLoss`  
- Accepts `alpha` and `beta` as per-class tensors `[C]`
- Applies `(1 - tversky_c)^gamma` focal weighting (γ ≥ 1) to down-weight easy classes
- Uses `class_weights` tensor for the final weighted sum
- Adipose/fascicle get: `alpha=0.2, beta=0.8` (higher recall bias)
- Background/connective get: `alpha=0.4, beta=0.6` (more balanced)
- Blood_vessel: same as background (too rare to trust)

**Impact:** Per-class FN penalty; focal term amplifies hard/missed regions.

#### C. New trainer: `nnUNetTrainerAdamEarlyStopping_TverskyPerClass`
**File:** `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdamEarlyStopping_Tversky.py`  
Combines:
- `PerClassFocalTverskyLoss` with class-specific α,β
- Corrected CE weights from tile distribution
- All existing early stopping / NaN protection / grad clipping from parent

This is the recommended trainer for Bio-Aegis generalization experiments.

---

### 🔲 Not yet implemented — future work

#### D. Adipose-targeted patch oversampling
Modify the nnUNet dataloader to maintain a per-class patch pool and explicitly
oversample patches containing adipose (class 2):
- 25% patches → must contain class 2
- 25% patches → must contain class 3
- 33% patches → any foreground (existing)
- 17% patches → random

This requires changes to `nnunetv2/training/dataloading/` and is more invasive.
Estimated gain: high, but implementation risk is higher than A/B/C.

#### E. Boundary loss for fascicle contours
Add a boundary Dice term weighted toward fascicle class:
```
L = L_tversky + λ_boundary * L_boundary_dice + λ_ce * L_CE
```
with `λ_boundary ≈ 0.1–0.3`.

Implementation: compute morphological boundary masks on-the-fly in the loss forward
pass (binary erosion/dilation via unfold or kornia). Small weight means no instability.

#### F. Domain oversampling via split_config
Add the two annotated Bio-Aegis training images (O11744, O19621) multiple times in
`split_config.json` to increase iSeg tile count from 40 to ~200 without new data.
Requires no code changes; just edit the JSON and retile.

#### G. Scale-normalise 10x → 40x before tiling
O22114 is 10× magnification — fascicle diameters are ~4× smaller in pixels vs. O21574.
Add a `scale_factor` field to split_config entries and apply bilinear upsampling in
`retile_all.py` before tiling. This aligns morphological scale across Bio-Aegis images.

---

## Experiment Matrix (recommended next runs)

| Trainer | Plan | Norm | Label |
|---|---|---|---|
| `nnUNetTrainerAdamEarlyStopping_TverskyPerClass` | `nnUNetPlansNoNorm` | none | `TV_PC_NoNorm` |
| `nnUNetTrainerAdamEarlyStopping_TverskyPerClass` | `nnUNetPlansHistoNorm` | histology | `TV_PC_Histo` |
| `nnUNetTrainerAdamEarlyStopping_TverskyPerClass` | `nnUNetPlansNyulNorm` | nyul | `TV_PC_Nyul` |

Run against existing CE and TV baselines to isolate the gain from the new loss.

---

## Changes Checklist

- [x] `documentation/improvement_plan_recall.md` — this file
- [x] `nnunetv2/training/loss/tversky.py` — `PerClassFocalTverskyLoss` added
- [x] `nnunetv2/training/loss/compound_losses.py` — `Tversky_and_CE_loss_PerClass` added
- [x] `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdamEarlyStopping.py` — CE weights corrected
- [x] `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdamEarlyStopping_Tversky.py` — CE weights corrected + `TverskyPerClass` trainer added
