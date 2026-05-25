# !/bin/bash

# Example baseline hyperparameters
BASE_DIR=results/fourway_1x1_left_right_turn_baselines
FR_H=700
FR_V=700

# # Priority (Horizontal)
# python intersection.py $BASE_DIR \
#     e=0 \
#     n_steps=3 \
#     n_rollouts_per_step=1 \
#     skip_stat_steps=0 \
#     av_frac=0 \
#     speed_mode=SPEED_MODE.all_checks \
#     priority=horizontal \
#     flow_rate_h=$FR_H \
#     flow_rate_v=$FR_V \
#     result_save=$BASE_DIR/eval_results/skip500_hpriority_flow${FR_H}x${FR_V}.csv
#     chain_lr=True \
#     use_poisson=True \
#     render

# Priority (Vertical)
python intersection.py $BASE_DIR \
    e=0 \
    n_steps=3 \
    n_rollouts_per_step=1 \
    skip_stat_steps=0 \
    av_frac=0 \
    speed_mode=SPEED_MODE.all_checks \
    priority=vertical \
    flow_rate_h=$FR_H \
    flow_rate_v=$FR_V \
    chain_lr=True \
    use_poisson=True \
    render=True \
    result_save=$BASE_DIR/eval_results/skip500_vpriority_flow${FR_H}x${FR_V}.csv
    
    
# # Traffic Signal with specified phase times
# PHASE_H=25 # Horizontal traffic signal phase length in seconds
# PHASE_V=25 # Vertical traffic signal phase length in seconds
# python intersection.py $BASE_DIR \
#     e=0 \
#     n_steps=3 \
#     n_rollouts_per_step=1 \
#     skip_stat_steps=500 \
#     av_frac=0 \
#     "'tl=($PHASE_H,$PHASE_V)'" \
#     yellow=0 \
#     flow_rate_h=$FR_H \
#     flow_rate_v=$FR_V \
#     use_poisson=True \
#     chain_lr=True \
#     render \
#     result_save=$BASE_DIR/eval_results/skip500_fixedtime_yellow0_no_lr_flow${FR_H}x${FR_V}.csv

# # Traffic Signal with MaxPressure
# MP_T_MIN=12 # Units are in seconds
# python intersection.py $BASE_DIR \
#     e=0 \
#     n_steps=3 \
#     n_rollouts_per_step=1 \
#     skip_stat_steps=0 \
#     av_frac=0 \
#     tl=MaxPressure \
#     mp_tmin=$MP_T_MIN \
#     yellow=3 \
#     flow_rate_h=$FR_H \
#     flow_rate_v=$FR_V \
#     use_poisson=True \
#     chain_lr=True \
#     render \
#     result_save=$BASE_DIR/eval_results/skip500_mpbest_yellow3_flow${FR_H}x${FR_V}.csv \
    

