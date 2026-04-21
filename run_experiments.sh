#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Runs 6 training experiments (3 normalisations × 2 losses),
# then runs inference for each, then reconstructs full-image predictions.
#
# Normalization is applied to FULL IMAGES during the tiling step (Phase 0)
# so that intensity statistics are computed over the whole slide, not per tile.
# All nnUNet plans use NoNormalization to avoid a second per-tile pass.
#
# Experiments:
#   CE_Histo   – DC+CE loss   + Histology pre-tiling norm (nnUNetPlansHistoNorm)
#   CE_Nyul    – DC+CE loss   + Nyul pre-tiling norm      (nnUNetPlansNyulNorm)
#   CE_NoNorm  – DC+CE loss   + No normalization          (nnUNetPlansNoNorm)
#   TV_Histo   – Tversky loss + Histology pre-tiling norm
#   TV_Nyul    – Tversky loss + Nyul pre-tiling norm
#   TV_NoNorm  – Tversky loss + No normalization
#
# Usage:
#   bash run_experiments.sh [--skip-tile] [--skip-train] [--skip-predict] [--fold F]
#   --skip-tile    : skip re-tiling and preprocessing (use existing nnUNet_raw data)
#   --skip-train   : skip training, only run inference + reconstruction
#   --skip-predict : skip inference + reconstruction, only tile + train
#   --fold F       : use fold F (default: 0)
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE="/home/moyasans/Documents/nnUNet"
export nnUNet_raw="$WORKSPACE/nnUNet_raw"
export nnUNet_preprocessed="$WORKSPACE/nnUNet_preprocessed"
export nnUNet_results="$WORKSPACE/nnUNet_results"

PYTHON="/home/moyasans/miniconda3/envs/segmentation/bin/python"
NNUNET_TRAIN="/home/moyasans/miniconda3/envs/segmentation/bin/nnUNetv2_train"
NNUNET_PREPROCESS="/home/moyasans/miniconda3/envs/segmentation/bin/nnUNetv2_preprocess"
NNUNET_PREDICT="/home/moyasans/miniconda3/envs/segmentation/bin/nnUNetv2_predict"

# Disable torch.compile to avoid 20-min JIT overhead on every experiment
export nnUNet_compile=0

DATASET_ID=1
CONFIG=2d
FOLD=0
IMAGES_TS="$nnUNet_raw/Dataset001_NerveMAVI/imagesTs"
NYUL_SCALE="$nnUNet_preprocessed/Dataset001_NerveMAVI/nyul_standard_scale.json"

SKIP_TILE=false
SKIP_TRAIN=false
SKIP_PREDICT=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-tile)    SKIP_TILE=true     ;;
        --skip-train)   SKIP_TRAIN=true    ;;
        --skip-predict) SKIP_PREDICT=true  ;;
        --fold)         shift; FOLD="$1"   ;;
    esac
done

# ── Experiment table: (label, trainer, plan) ─────────────────────────────────
# All plans use NoNormalization internally — normalization is applied to the
# full slide image before tiling (Phase 0), so nnUNet must NOT re-normalize
# the individual tiles during preprocessing.
declare -a LABELS=(
    "CE_Histo"
    "CE_Nyul"
    "CE_NoNorm"
    "TV_Histo"
    "TV_Nyul"
    "TV_NoNorm"
)
declare -a TRAINERS=(
    "nnUNetTrainerAdamEarlyStopping_LowLR"
    "nnUNetTrainerAdamEarlyStopping_LowLR"
    "nnUNetTrainerAdamEarlyStopping_LowLR"
    "nnUNetTrainerAdamEarlyStopping_Tversky"
    "nnUNetTrainerAdamEarlyStopping_Tversky"
    "nnUNetTrainerAdamEarlyStopping_Tversky"
)
declare -a PLANS=(
    "nnUNetPlansHistoNorm"
    "nnUNetPlansNyulNorm"
    "nnUNetPlansNoNorm"
    "nnUNetPlansHistoNorm"
    "nnUNetPlansNyulNorm"
    "nnUNetPlansNoNorm"
)

N=${#LABELS[@]}
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Helper: run inference + reconstruction for one experiment.
# Expects imagesTs to already hold the correctly normalised tiles.
# =============================================================================
run_predict_and_reconstruct() {
    local LABEL="$1" TRAINER="$2" PLAN="$3"
    local LOG_P="$LOG_DIR/predict_${LABEL}.log"
    local RESULTS_FOLD="$nnUNet_results/Dataset001_NerveMAVI/${TRAINER}__${PLAN}__${CONFIG}/fold_${FOLD}"
    local PRED_OUT="$RESULTS_FOLD/predictions"
    local RECON_OUT="$RESULTS_FOLD/reconstructed"
    mkdir -p "$PRED_OUT"

    echo "  [${LABEL}] predicting → $PRED_OUT"
    if "$NNUNET_PREDICT" \
           -d $DATASET_ID \
           -i "$IMAGES_TS" \
           -o "$PRED_OUT" \
           -c $CONFIG \
           -f $FOLD \
           -tr "$TRAINER" \
           -p "$PLAN" \
           > "$LOG_P" 2>&1; then
        echo "  [${LABEL}] ✓ inference done"
    else
        echo "  [${LABEL}] ✗ inference FAILED — check $LOG_P"
        return 1
    fi

    echo "  [${LABEL}] reconstructing…"
    "$PYTHON" "$WORKSPACE/reconstruct_predictions.py" \
        --predictions "$PRED_OUT" \
        --output      "$RECON_OUT" \
        --label       "$LABEL" \
        >> "$LOG_P" 2>&1 \
        && echo "  [${LABEL}] ✓ reconstruction done" \
        || echo "  [${LABEL}] ✗ reconstruction FAILED — check $LOG_P"

    echo "  [${LABEL}] post-processing reconstructed predictions…"
    "$PYTHON" "$WORKSPACE/postprocess_predictions.py" \
        --label "$LABEL" \
        --fold  "$FOLD" \
        >> "$LOG_P" 2>&1 \
        && echo "  [${LABEL}] ✓ postprocessing done" \
        || echo "  [${LABEL}] ✗ postprocessing FAILED — check $LOG_P"
}

# =============================================================================
# Main pipeline — one complete pass per normalization group.
#
# retile_all.py wipes ALL tile folders (imagesTr, imagesTs, labelsTr, labelsTs)
# on every run, so inference MUST happen before the next group retiles,
# otherwise test tiles would be from a different normalization than training.
#
# Order per group:
#   1. retile --norm X          → imagesTr + imagesTs hold X-normalised tiles
#   2. nnUNetv2_preprocess      → writes nnUNetPlansX_2d/  (training source)
#   3. nnUNetv2_train (×2)      → CE and TV experiments for this norm
#   4. nnUNetv2_predict (×2)    → reads imagesTs (still X-normalised)
#   5. reconstruct (×2)
#   → next group retiles (overwrites imagesTs – safe because inference ran)
# =============================================================================

declare -A NORM_PLAN=(
    [histology]="nnUNetPlansHistoNorm"
    [nyul]="nnUNetPlansNyulNorm"
    [none]="nnUNetPlansNoNorm"
)
# Experiment indices for each norm group (0=CE_Histo 1=CE_Nyul 2=CE_NoNorm
#                                         3=TV_Histo 4=TV_Nyul 5=TV_NoNorm)
declare -A NORM_INDICES=(
    [histology]="0 3"
    [nyul]="1 4"
    [none]="2 5"
)

TRAIN_FAILED=0

for NORM in histology nyul none; do
    PLAN="${NORM_PLAN[$NORM]}"
    INDICES="${NORM_INDICES[$NORM]}"

    echo ""
    echo "================================================================"
    echo "GROUP: $NORM  (plan: $PLAN)"
    echo "================================================================"

    # ── Step 1: Tiling + preprocessing ───────────────────────────────────────
    if [ "$SKIP_TILE" = false ]; then
        if [ "$NORM" = "nyul" ] && [ ! -f "$NYUL_SCALE" ]; then
            echo "  WARNING: Nyul scale file not found: $NYUL_SCALE"
            echo "  Run: python compute_nyul_scale.py  — skipping CE_Nyul + TV_Nyul."
            continue
        fi

        TILE_ARGS="--norm $NORM"
        [ "$NORM" = "nyul" ] && TILE_ARGS="$TILE_ARGS --scale-file $NYUL_SCALE"

        echo "  [tile] retile_all.py --norm $NORM"
        "$PYTHON" "$WORKSPACE/retile_all.py" $TILE_ARGS \
            > "$LOG_DIR/tile_${NORM}.log" 2>&1 \
            && echo "  [tile] ✓ done" \
            || { echo "  [tile] ✗ FAILED — check $LOG_DIR/tile_${NORM}.log"; exit 1; }

        echo "  [preprocess] $PLAN"
        "$NNUNET_PREPROCESS" -d $DATASET_ID -plans_name "$PLAN" -c $CONFIG -np 4 \
            > "$LOG_DIR/preprocess_${NORM}.log" 2>&1 \
            && echo "  [preprocess] ✓ done" \
            || { echo "  [preprocess] ✗ FAILED — check $LOG_DIR/preprocess_${NORM}.log"; exit 1; }
    fi

    # ── Step 2: Training ──────────────────────────────────────────────────────
    if [ "$SKIP_TRAIN" = false ]; then
        for i in $INDICES; do
            LABEL="${LABELS[$i]}"
            TRAINER="${TRAINERS[$i]}"
            LOG="$LOG_DIR/train_${LABEL}.log"
            echo "  [train] $LABEL → $LOG"
            if "$NNUNET_TRAIN" $DATASET_ID $CONFIG $FOLD -tr "$TRAINER" -p "$PLAN" \
                   > "$LOG" 2>&1; then
                echo "  [train] ✓ $LABEL done"
            else
                echo "  [train] ✗ $LABEL FAILED — check $LOG"
                TRAIN_FAILED=$((TRAIN_FAILED + 1))
            fi
        done
        if [ "$TRAIN_FAILED" -gt 0 ]; then
            echo "  WARNING: $TRAIN_FAILED training job(s) failed in group $NORM."
        fi
    fi

    # ── Step 3: Inference + reconstruction ───────────────────────────────────
    # imagesTs still holds the $NORM-normalised tiles written in Step 1.
    if [ "$SKIP_PREDICT" = false ]; then
        for i in $INDICES; do
            LABEL="${LABELS[$i]}"
            TRAINER="${TRAINERS[$i]}"
            run_predict_and_reconstruct "$LABEL" "$TRAINER" "$PLAN" || true
        done
    fi

done



echo ""
echo "================================================================"
echo "All done."
echo "Reconstructed images are in:"
for (( i=0; i<N; i++ )); do
    LABEL="${LABELS[$i]}"
    TRAINER="${TRAINERS[$i]}"
    PLAN="${PLANS[$i]}"
    echo "  [${LABEL}] $nnUNet_results/Dataset001_NerveMAVI/${TRAINER}__${PLAN}__${CONFIG}/fold_${FOLD}/reconstructed/"
done
echo "Open visualize_predictions.ipynb to compare results."
echo "================================================================"
