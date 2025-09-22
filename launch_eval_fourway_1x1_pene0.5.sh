#!/bin/bash

# Example evaluation hyperparameters
EXP_DIR=results/fourway_2x1_penetration0.5
CKPT=0
FR_H=850 # Horizontal flow rate in vehicles/hour
FR_V=850 # Vertical flow rate
N_ROWS=2
N_COLS=1
RESULT_SAVE_PATH=$EXP_DIR/eval_results/e0_2x1_skip500_flow850x850.csv
DIRECTION="4way"

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
    render