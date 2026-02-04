import torch
import numpy as np
import glob
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def find_best_model(base_folder):
    """
    Find the best model from a set of model folders by comparing their losses.
    
    Args:
        base_folder: Path to the base folder pattern (e.g., 'models/20260126_131210')
                    All folders matching this pattern + '*' will be searched.
    
    Returns:
        tuple: (best_model_folder, best_loss) - Path to the best model folder and its loss
    """
    base_folder = Path(base_folder)
    
    # Find all model folders matching the base path pattern (e.g., models/20260106_042356*)
    base_folder_str = str(base_folder)
    matching_folders = sorted(glob.glob(base_folder_str + '*'))
    matching_folders = [Path(f) for f in matching_folders if Path(f).is_dir()]
    
    if len(matching_folders) == 0:
        raise ValueError(f"No model folders found matching pattern: {base_folder_str}*")
    
    print(f"Found {len(matching_folders)} model folders:")
    for f in matching_folders:
        print(f"  {f}")
    
    # Compute the minimum loss for each model folder
    model_losses = []
    for folder in matching_folders:
        opt_path = folder / 'model_opt.pt'
        if opt_path.exists():
            opt_data = torch.load(opt_path, map_location='cpu', weights_only=False)
            min_loss = min(opt_data['loss'])
            if ~np.isnan(min_loss) and ~np.isinf(min_loss):
                model_losses.append((folder, min_loss))
            print(f"  {folder.name}: min loss = {min_loss:.6f}")
        else:
            print(f"  {folder.name}: model_opt.pt not found, skipping")
    
    if len(model_losses) == 0:
        raise ValueError("No valid models found with model_opt.pt")
    
    # Create scatter plot of model losses
    losses = [loss for _, loss in model_losses]
    plt.figure(figsize=(6, 8))
    plt.scatter([0] * len(losses), losses, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    plt.ylabel('Minimum Loss')
    plt.title(f'Model Losses ({len(model_losses)} runs)')
    plt.xticks([])
    plt.xlim(-0.5, 0.5)
    
    # Highlight the best model
    best_idx = np.argmin(losses)
    plt.scatter([0], [losses[best_idx]], s=120, c='red', marker='*', zorder=5, label='Best')
    plt.legend()
    
    # Save plot to the base folder (without suffix)
    loss_comparison_path = base_folder.parent.parent / Path('figures') / (base_folder.name + '_loss_comparison.png')
    loss_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLoss comparison plot saved to: {loss_comparison_path}")
    
    # Select the model with the lowest loss
    model_folder, best_loss = model_losses[best_idx]
    print(f"\nSelected best model: {model_folder} (loss = {best_loss:.6f})")
    
    return model_folder, best_loss


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python find_best_model.py <base_folder>")
        print("Example: python find_best_model.py models/20260126_131210")
        sys.exit(1)
    
    base_folder = Path(sys.argv[1])
    best_model_folder, best_loss = find_best_model(base_folder)
    print(f"\nBest model folder: {best_model_folder}")
    print(f"Best loss: {best_loss:.6f}")
