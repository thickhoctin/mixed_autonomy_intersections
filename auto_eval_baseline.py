#!/usr/bin/env python3
from pathlib import Path
import subprocess
import itertools

BASE_DIR = Path("results/fourway_1x1_left_right_turn_baselines")

FLOW_PAIRS = [
    (700, 700),
    (850, 850),
    (850, 700),
    (850, 550),
    (850, 400),
    (700, 850),
    (550, 850),
    (400, 850),
    (1000, 850),
    (1000, 700),
    (1000, 550),
    (1000, 400),
    (850, 1000),
    (700, 1000),
    (550, 1000),
    (400, 1000),
]

BASELINES = [
    {
        "name": "vpriority",
        "extra_args": [
            "e=0",
            "n_steps=3",
            "n_rollouts_per_step=1",
            "skip_stat_steps=0",
            "av_frac=0",
            "speed_mode=SPEED_MODE.all_checks",
            "priority=vertical",
            "chain_lr=True",
            "use_poisson=True",
            "render=False",
        ],
        "result_template": "eval_results/skip500_vpriority_flow{h}x{v}.csv",
    },
    {
        "name": "hpriority",
        "extra_args": [
            "e=0",
            "n_steps=3",
            "n_rollouts_per_step=1",
            "skip_stat_steps=0",
            "av_frac=0",
            "speed_mode=SPEED_MODE.all_checks",
            "priority=horizontal",
            "chain_lr=True",
            "use_poisson=True",
            "render=False",
        ],
        "result_template": "eval_results/skip500_hpriority_flow{h}x{v}.csv",
    },
    {
        "name": "signalbest_yellow0",
        "extra_args": [
            "e=0",
            "n_steps=3",
            "n_rollouts_per_step=1",
            "skip_stat_steps=500",
            "av_frac=0",
            "'tl=(25,25)'",
            "yellow=0",
            "chain_lr=True",
            "use_poisson=True",
            "render=False",
        ],
        "result_template": "eval_results/skip500_fixedtime_yellow0_no_lr_flow{h}x{v}.csv",
    },
    {
        "name": "mpbest_yellow3",
        "extra_args": [
            "e=0",
            "n_steps=3",
            "n_rollouts_per_step=1",
            "skip_stat_steps=0",
            "av_frac=0",
            "tl=MaxPressure",
            "mp_tmin=12",
            "yellow=3",
            "chain_lr=True",
            "use_poisson=True",
            "render=False",
        ],
        "result_template": "eval_results/skip500_mpbest_yellow3_flow{h}x{v}.csv",
    },
]

def run_one(flow_h, flow_v, baseline):
    result_save = BASE_DIR / baseline["result_template"].format(h=flow_h, v=flow_v)
    result_save.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "intersection.py",
        str(BASE_DIR),
        *baseline["extra_args"],
        f"flow_rate_h={flow_h}",
        f"flow_rate_v={flow_v}",
        f"result_save={result_save.as_posix()}",
    ]

    print("\n" + "=" * 100)
    print(f"Running {baseline['name']} for flow {flow_h} x {flow_v}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base dir not found: {BASE_DIR}")

    total = len(FLOW_PAIRS) * len(BASELINES)
    idx = 0

    for flow_h, flow_v in FLOW_PAIRS:
        for baseline in BASELINES:
            idx += 1
            print(f"\n[{idx}/{total}]")
            run_one(flow_h, flow_v, baseline)

if __name__ == "__main__":
    main()