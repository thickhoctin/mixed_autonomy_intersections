#!/bin/bash

# Example evaluation hyperparameters
EXP_DIR=results/twoway_2x1_penetration0.333
CKPT=200
FR_H=850 # Horizontal flow rate in vehicles/hour
FR_V=700 # Vertical flow rate
N_ROWS=3
N_COLS=3
RESULT_SAVE_PATH=$EXP_DIR/eval_results/e165_3x3_skip500_flow850x700.csv

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
