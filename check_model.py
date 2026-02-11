import torch
import os
import glob
import sys

def inspect_latest_model():
    # 1. Define the directory (Note the space in 'flow 700x700')
    # Using raw string (r"...") to avoid issues with special characters
    target_dir = r"results/fourway_1x1_penetration0.5_turn_adam_ppo_transformer_11.02/models/flow_700x700"

    print(f"Searching in: {target_dir}")

    # 2. Find all .pth files
    # Check if directory exists first
    if not os.path.exists(target_dir):
        print(f"❌ Error: Directory not found.\n   -> {target_dir}")
        return

    # Get list of all .pth files
    list_of_files = glob.glob(os.path.join(target_dir, "*.pth"))

    if not list_of_files:
        print("❌ Error: No .pth files found in that directory.")
        return

    # 3. Get the latest file (based on creation time)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"✅ Found latest model: {os.path.basename(latest_file)}")
    print(f"   Path: {latest_file}\n")

    # 4. Load the model
    try:
        # map_location='cpu' ensures it loads even if you don't have a GPU active
        checkpoint = torch.load(latest_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 5. Extract the State Dictionary
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Assume the dict is the weights
    elif isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    
    if state_dict is None:
        print("❌ Could not extract state_dict from checkpoint.")
        return

    # 6. Search for Transformer Layers
    transformer_keys = []
    print(f"--- Inspecting {len(state_dict)} Layers ---\n")

    for key in state_dict['net'].keys():
        # Check for keywords specific to your implementation
        if "transformer" in key or "attention" in key or "attn" in key:
            transformer_keys.append(key)

    # 7. Print Results
    if len(transformer_keys) > 0:
        print(f"🎉 CONFIRMED: Transformer Architecture Detected!")
        print(f"Found {len(transformer_keys)} layers related to Transformers.")
        print("\nSample Transformer Keys found:")
        # Print first 5 and last 5 to give a good overview
        for k in transformer_keys[:5]:
            print(f"  • {k}")
        if len(transformer_keys) > 5:
            print("  ... (others omitted) ...")
    else:
        print("⚠️ WARNING: No Transformer layers found.")
        print("Here are the first 10 layers found (likely MLP/CNN):")
        for k in list(state_dict.keys())[:10]:
            print(f"  • {k}")

if __name__ == "__main__":
    inspect_latest_model()