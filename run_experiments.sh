#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Runs 6 parallel training experiments (3 normalisations × 2 losses),
# then runs inference for each, then reconstructs full-image predictions.
#
# Experiments:
#   CE_Histo   – DC+CE loss + HistologyNormalization  (nnUNetPlans)
#   CE_Nyul    – DC+CE loss + NyulNormalization        (nnUNetPlansNyul)
#   CE_NoNorm  – DC+CE loss + NoNormalization          (nnUNetPlansNoNorm)
#   TV_Histo   – Tversky loss + HistologyNormalization
#   TV_Nyul    – Tversky loss + NyulNormalization
#   TV_NoNorm  – Tversky loss + NoNormalization
#
# Usage:
#   bash run_experiments.sh [--skip-train] [--skip-predict] [--fold F]
#   --skip-train   : skip training, only run inference + reconstruction
#   --skip-predict : skip inference + reconstruction, only train
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
NNUNET_PREDICT="/home/moyasans/miniconda3/envs/segmentation/bin/nnUNetv2_predict"

# Disable torch.compile to avoid 20-min JIT overhead on every experiment
export nnUNet_compile=0

DATASET_ID=1
CONFIG=2d
FOLD=0
IMAGES_TS="$nnUNet_raw/Dataset001_NerveMAVI/imagesTs"

SKIP_TRAIN=false
SKIP_PREDICT=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --skip-train)   SKIP_TRAIN=true    ;;
        --skip-predict) SKIP_PREDICT=true  ;;
        --fold)         shift; FOLD="$1"   ;;
    esac
done

# ── Experiment table: (label, trainer, plan) ─────────────────────────────────
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
    "nnUNetPlans"
    "nnUNetPlansNyul"
    "nnUNetPlansNoNorm"
    "nnUNetPlans"
    "nnUNetPlansNyul"
    "nnUNetPlansNoNorm"
)

N=${#LABELS[@]}
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# PHASE 1 – TRAINING (sequential – one GPU, avoid OOM)
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo "================================================================"
    echo "PHASE 1: Running $N training experiments sequentially"
    echo "================================================================"

    FAILED=0
    for (( i=0; i<N; i++ )); do
        LABEL="${LABELS[$i]}"
        TRAINER="${TRAINERS[$i]}"
        PLAN="${PLANS[$i]}"
        LOG="$LOG_DIR/train_${LABEL}.log"

        echo "  [${LABEL}] training → $LOG"
        if "$NNUNET_TRAIN" $DATASET_ID $CONFIG $FOLD \
               -tr "$TRAINER" -p "$PLAN" \
               > "$LOG" 2>&1; then
            echo "  [${LABEL}] ✓ done"
        else
            echo "  [${LABEL}] ✗ FAILED — check $LOG"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -gt 0 ]; then
        echo ""
        echo "WARNING: $FAILED training job(s) failed. Inference will still run for completed ones."
    fi
fi

# =============================================================================
# PHASE 2 – INFERENCE (parallel)
# =============================================================================
if [ "$SKIP_PREDICT" = false ]; then
    echo ""
    echo "================================================================"
    echo "PHASE 2: Running inference for $N experiments in parallel"
    echo "================================================================"

    for (( i=0; i<N; i++ )); do
        LABEL="${LABELS[$i]}"
        TRAINER="${TRAINERS[$i]}"
        PLAN="${PLANS[$i]}"
        LOG="$LOG_DIR/predict_${LABEL}.log"

        # nnUNet output folder for this experiment
        RESULTS_FOLD="$nnUNet_results/Dataset001_NerveMAVI/${TRAINER}__${PLAN}__${CONFIG}/fold_${FOLD}"
        PRED_OUT="$RESULTS_FOLD/predictions"
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
               > "$LOG" 2>&1; then
            echo "  [${LABEL}] ✓ inference done"
        else
            echo "  [${LABEL}] ✗ FAILED — check $LOG"
        fi
    done

    # ==========================================================================
    # PHASE 3 – RECONSTRUCTION (tile predictions → full images)
    # ==========================================================================
    echo ""
    echo "================================================================"
    echo "PHASE 3: Reconstructing full images from tile predictions"
    echo "================================================================"

    for (( i=0; i<N; i++ )); do
        LABEL="${LABELS[$i]}"
        TRAINER="${TRAINERS[$i]}"
        PLAN="${PLANS[$i]}"
        PRED_OUT="$nnUNet_results/Dataset001_NerveMAVI/${TRAINER}__${PLAN}__${CONFIG}/fold_${FOLD}/predictions"
        RECON_OUT="$nnUNet_results/Dataset001_NerveMAVI/${TRAINER}__${PLAN}__${CONFIG}/fold_${FOLD}/reconstructed"

        if [ -d "$PRED_OUT" ] && [ "$(ls "$PRED_OUT"/*.tif 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [${LABEL}] reconstructing …"
            "$PYTHON" "$WORKSPACE/reconstruct_predictions.py" \
                --predictions "$PRED_OUT" \
                --output      "$RECON_OUT" \
                --label       "$LABEL"
        else
            echo "  [${LABEL}] SKIP — no predictions found in $PRED_OUT"
        fi
    done
fi

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
