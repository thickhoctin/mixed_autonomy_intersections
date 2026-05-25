import argparse
import subprocess
import torch
from pathlib import Path


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

# Best checkpoints selected from the notebook charts/table
BEST_CHECKPOINTS = {
    # "fourway_1x1_penetration0.5_turn_adam_ppo_12.12": 220,
    # "fourway_1x1_penetration0.5_turn_adam_ppo_17.12": 300,
    # "fourway_1x1_penetration0.5_turn_adam_ppo_transformer_02.02": 65,
    "fourway_1x1_penetration0.5_turn_adam_ppo_transformer_13.02": 350,
}


def infer_grid_from_name(name: str):
    if "1x1" in name:
        return 1, 1
    if "2x1" in name:
        return 2, 1
    if "3x3" in name:
        return 3, 3
    return 1, 1


def build_experiment_dirs(results_root: Path):
    exp_dirs = []
    for exp_name in sorted(BEST_CHECKPOINTS.keys()):
        exp_dir = results_root / exp_name
        if exp_dir.exists():
            exp_dirs.append(exp_dir)
    return exp_dirs


def ensure_checkpoint_for_flow_rate(exp_dir: Path, ckpt: int, flow_h: int, flow_v: int):
    """
    Copy checkpoint from 700x700 flow rate to target flow rate if needed.
    Models are stored in models/flow_{flow_h}x{flow_v}/ directories.
    """
    # Only copy if flow rate is different from training flow rate (700x700)
    if flow_h == 700 and flow_v == 700:
        return
    
    # Source: trained model at 700x700
    src_flow_dir = exp_dir / "models" / "flow_700x700"
    src_ckpt_path = src_flow_dir / f"model-{ckpt}.pth"
    
    # Target: evaluation flow rate directory
    tgt_flow_dir = exp_dir / "models" / f"flow_{flow_h}x{flow_v}"
    tgt_ckpt_path = tgt_flow_dir / f"model-{ckpt}.pth"
    
    # If target checkpoint already exists, skip
    if tgt_ckpt_path.exists():
        print(f"Checkpoint already exists at {tgt_ckpt_path}")
        return
    
    # Check if source checkpoint exists
    if not src_ckpt_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt_path}")
    
    # Create target directory
    tgt_flow_dir.mkdir(parents=True, exist_ok=True)
    
    # Load source checkpoint and copy only the 'net' part
    print(f"Copying checkpoint from {src_ckpt_path} to {tgt_ckpt_path}")
    model_dict = torch.load(src_ckpt_path)
    new_model_dict = dict(net=model_dict['net'], step=model_dict['step'])
    torch.save(new_model_dict, tgt_ckpt_path)
    print(f"Checkpoint copied successfully")


def run_one_evaluation(exp_dir: Path, ckpt: int, flow_h: int, flow_v: int, dry_run: bool):
    n_rows, n_cols = infer_grid_from_name(exp_dir.name)
    
    # Ensure checkpoint exists for target flow rate
    if not dry_run:
        ensure_checkpoint_for_flow_rate(exp_dir, ckpt, flow_h, flow_v)
    loss = 5    
    result_name = f"e{ckpt}_1x1_skip_500_loss{loss}_flow{flow_h}x{flow_v}.csv"
    result_save = exp_dir / "eval_results" / result_name
    vehicle_save = exp_dir / "vehicle_info" / result_name.replace(".csv", "_vehicle_info.csv")
    if "transformer" in exp_dir.name:
        use_attention = True
    else:
        use_attention = False
    cmd = [
        "python3",
        "intersection.py",
        exp_dir.as_posix(),
        f"e={ckpt}",
        f"n_rows={n_rows}",
        f"n_cols={n_cols}",
        "n_steps=10",
        "n_rollouts_per_step=1",
        "skip_stat_steps=500",
        f"flow_rate_h={flow_h}",
        f"flow_rate_v={flow_v}",
        f"result_save={result_save.as_posix()}",
        f"vehicle_info_save={vehicle_save.as_posix()}",
        "use_ray=False",
        "use_poisson=True",
        f"use_attention={use_attention}",
        f"obs_packet_loss={loss/100 if loss > 0 else 0}",
    ]

    print("=" * 100)
    print(f"Experiment : {exp_dir.as_posix()}")
    print(f"Checkpoint  : {ckpt}")
    print(f"Flow rates  : {flow_h} x {flow_v}")
    print(f"Result file : {result_save.as_posix()}")
    print(f"Vehicle info: {vehicle_save.as_posix()}")

    if dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return

    exp_dir.joinpath("eval_results").mkdir(parents=True, exist_ok=True)
    exp_dir.joinpath("vehicle_info").mkdir(parents=True, exist_ok=True)

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment folder names to evaluate",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"Results dir not found: {results_root}")

    exp_dirs = build_experiment_dirs(results_root)
    if args.experiments:
        wanted = set(args.experiments)
        exp_dirs = [exp_dir for exp_dir in exp_dirs if exp_dir.name in wanted]

    if not exp_dirs:
        raise RuntimeError("No matching experiments found to evaluate.")

    total_jobs = len(exp_dirs) * len(FLOW_PAIRS)
    print(f"Found {len(exp_dirs)} experiments.")
    print(f"Planning {total_jobs} evaluation jobs.")

    job_index = 0
    for exp_dir in exp_dirs:
        ckpt = BEST_CHECKPOINTS[exp_dir.name]
        for flow_h, flow_v in FLOW_PAIRS:
            job_index += 1
            print(f"\n[{job_index}/{total_jobs}]")
            run_one_evaluation(exp_dir, ckpt, flow_h, flow_v, args.dry_run)


if __name__ == "__main__":
    main()