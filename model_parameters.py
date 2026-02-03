import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys

def load_and_analyze_model(model_dir, model_step):
    """Load and analyze a specific model checkpoint"""
    
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / f'model-{model_step}.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints:")
        for cp in sorted(model_dir.glob('model-*.pth')):
            step = cp.stem.split('-')[1]
            print(f"  - Model {step}")
        return None
    
    print(f"\n{'='*80}")
    print(f"LOADING MODEL CHECKPOINT - MODEL {model_step}")
    print(f"{'='*80}")
    print(f"Path: {checkpoint_path}\n")
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check what's in the checkpoint
    print("Checkpoint type:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
    
    # Extract model - handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            model_state_dict = checkpoint['net']
            print("\nUsing checkpoint['net']")
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            print("\nUsing checkpoint['model']")
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            print("\nUsing checkpoint['state_dict']")
        else:
            # Assume the dict itself is the state dict
            model_state_dict = checkpoint
            print("\nUsing checkpoint as state_dict")
    else:
        # Checkpoint is already the model state dict
        model_state_dict = checkpoint
    
    print(f"\n{'='*80}")
    print("MODEL PARAMETERS")
    print(f"{'='*80}")
    
    total_params = 0
    param_data = []
    
    # Iterate through parameters
    for name, param in model_state_dict.items():
        # Only process weight and bias parameters
        if 'weight' in name or 'bias' in name:
            num_params = param.numel()
            total_params += num_params
            
            stats = {
                'Parameter': name,
                'Shape': str(list(param.shape)),
                'Num_Params': num_params,
                'Mean': f"{param.mean().item():.6f}",
                'Std': f"{param.std().item():.6f}",
                'Min': f"{param.min().item():.6f}",
                'Max': f"{param.max().item():.6f}",
            }
            param_data.append(stats)
            
            print(f"\n{name}:")
            print(f"  Shape:      {list(param.shape)}")
            print(f"  Num params: {num_params:,}")
            print(f"  Mean:       {param.mean().item():.6f}")
            print(f"  Std:        {param.std().item():.6f}")
            print(f"  Range:      [{param.min().item():.6f}, {param.max().item():.6f}]")
    
    print(f"\n{'='*80}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"{'='*80}\n")
    
    # Create DataFrame
    df = pd.DataFrame(param_data)
    
    # Save to CSV
    # output_csv = model_dir.parent / f'model_params_model_{model_step}.csv'
    # df.to_csv(output_csv, index=False)
    # print(f"✓ Exported to: {output_csv}\n")
    
    return df, model_state_dict

def main():
    # Configuration
    model_dir = 'results/fourway_1x1_penetration0.5_turn_adam_ppo_27.11/models/flow_700x700'
    
    # Get model step from command line or prompt
    if len(sys.argv) > 1:
        model_step = sys.argv[1]
    else:
        print("\nAvailable checkpoints:")
        model_dir_path = Path(model_dir)
        checkpoints = sorted(model_dir_path.glob('model-*.pth'))
        
        if not checkpoints:
            print(f"❌ No model checkpoints found in {model_dir_path}")
            return
        
        for cp in checkpoints:
            step = cp.stem.split('-')[1]
            print(f"  - Model {step}")
        
        model_step = input("\nEnter model number: ").strip()
    
    # Load and analyze
    result = load_and_analyze_model(model_dir, model_step)
    
    if result is not None:
        df, model_state_dict = result
        print("\nSummary Table:")
        print(df.to_string(index=False))
        
        # Additional analysis
        print(f"\n{'='*80}")
        print("ADDITIONAL ANALYSIS")
        print(f"{'='*80}")
        
        # Check for potential issues
        print("\n1. Checking for very small weights (potential initialization issues):")
        for name, param in model_state_dict.items():
            if 'weight' in name:
                std_val = param.std().item()
                if std_val < 0.001:
                    print(f"  ⚠️  {name}: std={std_val:.6f} (too small!)")
        
        print("\n2. Checking for very large weights (potential instability):")
        for name, param in model_state_dict.items():
            if 'weight' in name:
                max_val = param.abs().max().item()
                if max_val > 10.0:
                    print(f"  ⚠️  {name}: max={max_val:.6f} (too large!)")
        
        print("\n3. Layer sizes:")
        for name, param in model_state_dict.items():
            if 'weight' in name and len(param.shape) >= 2:
                in_features = param.shape[1]
                out_features = param.shape[0]
                print(f"  {name}: {in_features} → {out_features}")

if __name__ == '__main__':
    main()