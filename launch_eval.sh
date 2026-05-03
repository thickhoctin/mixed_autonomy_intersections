#!/bin/bash

# Example evaluation hyperparameters
EXP_DIR=results/fourway_1x1_penetration0.333_turn_adam_ppo_15.12
CKPT=260
FR_H=700 # Horizontal flow rate in vehicles/hour
FR_V=700 # Vertical flow rate
N_ROWS=1
N_COLS=1
RESULT_SAVE_PATH=$EXP_DIR/eval_results/e260_1x1_skip500_flow700x700.csv
VEHICLE_INFO_SAVE_PATH=$EXP_DIR/vehicle_info/e260_1x1_skip500_flow700x700_vehicle_info.csv
mkdir -p "$EXP_DIR/eval_results" "$EXP_DIR/vehicle_info" \
    && echo "Created directories for evaluation results and vehicle info in $EXP_DIR"
python3 intersection.py $EXP_DIR \
    e=$CKPT \
    n_rows=$N_ROWS \
    n_cols=$N_COLS \
    n_steps=10 \
    n_rollouts_per_step=1 \
    skip_stat_steps=500 \
    flow_rate_h=$FR_H \
    flow_rate_v=$FR_V \
    result_save=$RESULT_SAVE_PATH \
    vehicle_info_save=$VEHICLE_INFO_SAVE_PATH \
    use_ray=False \
    
    
