#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="/home/moyasans/Documents/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/moyasans/Documents/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/moyasans/Documents/nnUNet/nnUNet_results"

TRAINER="nnUNetTrainerAdamEarlyStopping_LowLR"
PLANS="nnUNetPlansMultiCh"

echo "Starting training with trainer: $TRAINER  plans: $PLANS"

exec /home/moyasans/miniconda3/envs/segmentation/bin/nnUNetv2_train \
    1 2d 0 -tr "$TRAINER" -p "$PLANS"
