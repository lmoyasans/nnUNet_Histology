#!/usr/bin/env bash
set -euo pipefail

###############################################
#### SET PARAMETERS FOR THE NN EXECUTION ######

USE_TRAINING=${USE_TRAINING:-true}
FOLDS=${FOLDS:-0}
TRAINER=${TRAINER:-"nnUNetTrainerAdamEarlyStopping"}
IMAGE_DIM=${IMAGE_DIM:-"2d"}

###############################################
###############################################

# Run the dataset preparation script
#python3 prepareDataset.py

export nnUNet_raw="/home/smu/work/workspace/nnUNet_raw"
export nnUNet_preprocessed="/home/smu/work/workspace/nnUNet_preprocessed"
export nnUNet_results="/home/smu/work/workspace/nnUNet_results"

# Install nnUNet_Histology in editable mode
echo 
echo "################################################"
echo "########## INSTALLING NNUNET LIBARIES ##########"
echo "################################################"
echo 

pip install -q -e nnUNet_Histology/

echo
echo "############################################################"
echo "############## STARTING NETWORK EXECUTION  #################"
echo "############################################################"
echo

# Determine dataset folder name under nnUNet_raw (expect exactly one)
DATASET_DIRS=("$nnUNet_raw"/*)
DATASET_NAME=$(basename "${DATASET_DIRS[0]}")

if [ "$USE_TRAINING" = true ]; then
    echo "Training is enabled."
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
    nnUNetv2_train 1 "$IMAGE_DIM" "$FOLDS" -tr "$TRAINER" --npz
    nnUNetv2_predict \
          -i "$nnUNet_raw/$DATASET_NAME/imagesTs" \
          -o "$nnUNet_results/$DATASET_NAME/predictions_${TRAINER}_${IMAGE_DIM}_${FOLDS}Folds" \
          -d 1 \
          -c "$IMAGE_DIM" \
          -tr "$TRAINER" \
          -f "$FOLDS" \
          -npp 1 \
          -nps 1 \
          --save_probabilities
    
else
    echo "Training is disabled."
    CHECKPOINT="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__${IMAGE_DIM}/fold_${FOLDS}/checkpoint_final.pth"
    if [ -f "$CHECKPOINT" ]; then
        echo "Found trained model checkpoint: $CHECKPOINT"
        echo "Starting segmentation prediction"
        nnUNetv2_predict \
          -i "$nnUNet_raw/$DATASET_NAME/imagesTs" \
          -o "$nnUNet_results/$DATASET_NAME/predictions_${TRAINER}_${IMAGE_DIM}_${FOLDS}Folds" \
          -d 1 \
          -c "$IMAGE_DIM" \
          -tr "$TRAINER" \
          -f "$FOLDS" \
          -npp 1 \
          -nps 1 \
          --save_probabilities
    else
        echo "Error: No completed checkpoint found at: $CHECKPOINT" >&2
        echo "Expected trainer='$TRAINER', config='$IMAGE_DIM', fold='$FOLDS'." >&2
        echo "Either enable training (USE_TRAINING=true) or provide a matching pretrained model." >&2
        exit 1
    fi
fi
