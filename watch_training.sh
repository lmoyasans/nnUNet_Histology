#!/usr/bin/env bash
# watch_training.sh  – one-line status per experiment
# Usage:  watch -n 30 bash watch_training.sh

RESULTS="/home/moyasans/Documents/nnUNet/nnUNet_results/Dataset001_NerveMAVI"
LOGS="/home/moyasans/Documents/nnUNet/logs"

declare -a LABELS=(CE_Histo CE_Nyul CE_NoNorm TV_Histo TV_Nyul TV_NoNorm)
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

echo "$(date '+%H:%M:%S')  Training status"
echo "────────────────────────────────────────────────────────────"
printf "%-12s  %-6s  %-9s  %-9s  %-10s  %s\n" "Experiment" "Epoch" "tr_loss" "val_loss" "PseudoDice" "Status"
echo "────────────────────────────────────────────────────────────"

for (( i=0; i<${#LABELS[@]}; i++ )); do
    LABEL="${LABELS[$i]}"
    TRAINER="${TRAINERS[$i]}"
    PLAN="${PLANS[$i]}"
    LOG_DIR="$RESULTS/${TRAINER}__${PLAN}__2d/fold_0"
    TLOG=$(ls -t "$LOG_DIR"/training_log_*.txt 2>/dev/null | head -1)

    if [ -z "$TLOG" ]; then
        printf "%-12s  %-6s\n" "$LABEL" "not started"
        continue
    fi

    EPOCH=$(grep ": Epoch [0-9]" "$TLOG" | tail -1 | grep -oP 'Epoch \K[0-9]+')
    TR=$(grep "train_loss" "$TLOG" | tail -1 | grep -oP 'train_loss \K[-\d.]+')
    VAL=$(grep "val_loss " "$TLOG" | grep -v "improved\|No improvement" | tail -1 | grep -oP 'val_loss \K[-\d.]+')
    DICE_LINE=$(grep "Pseudo dice" "$TLOG" | tail -1)
    DICE_MEAN=$(echo "$DICE_LINE" | python3 -c "
import sys, re, numpy as np
line = sys.stdin.read()
m = re.search(r'\[([^\]]+)\]', line)
if m:
    vals = [float(v) for v in re.findall(r'-?\d+\.\d+', m.group(1))]
    vals = [v for v in vals if 0 <= v <= 1]
    print(f'{np.mean(vals):.4f}' if vals else '-')
else:
    print('-')
" 2>/dev/null)

    if grep -q "The best model checkpoint" "$TLOG" 2>/dev/null; then
        STATUS="✓ done"
    elif [[ $(find "$LOG_DIR" -name "training_log_*.txt" -mmin -2 2>/dev/null | wc -l) -gt 0 ]]; then
        STATUS="▶ running"
    else
        STATUS="waiting"
    fi

    printf "%-12s  %-6s  %-9s  %-9s  %-10s  %s\n" \
        "$LABEL" "${EPOCH:--}" "${TR:--}" "${VAL:--}" "${DICE_MEAN:--}" "$STATUS"
done

echo "────────────────────────────────────────────────────────────"
echo "Active: $(ps aux | grep '[n]nUNetv2_train' | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}' | head -1)"
echo "Main:   $(tail -1 "$LOGS/run_experiments_main.log" 2>/dev/null)"
