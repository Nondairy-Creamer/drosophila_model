import torch
import numpy as np
import re
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
NEAR_ZERO_THRESHOLD = 1e-6


def find_best_models(models_dir=MODELS_DIR):
    """
    Scan the models directory, group model folders by their tag/timestamp prefix
    (stripping any _gpuN_runN suffix), and for each group find the folder with
    the lowest valid loss.

    Losses that are NaN, inf, or within 1e-6 of zero are ignored.

    Returns:
        dict: {timestamp_prefix: (best_folder, best_loss)}
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise ValueError(f"Models directory not found: {models_dir}")

    # Collect all subdirectories (skip hidden directories)
    all_folders = sorted(
        f for f in models_dir.iterdir()
        if f.is_dir() and not f.name.startswith('.')
    )

    if not all_folders:
        raise ValueError(f"No model folders found in {models_dir}")

    # Group by stripping the _gpuN_runN suffix (if present) from the folder name
    groups = defaultdict(list)
    for folder in all_folders:
        prefix = re.sub(r'_gpu\d+_run\d+$', '', folder.name)
        groups[prefix].append(folder)

    results = {}

    for prefix in sorted(groups):
        folders = groups[prefix]
        print(f"\n{'='*60}")
        print(f"Group: {prefix}  ({len(folders)} run(s))")
        print(f"{'='*60}")

        model_losses = []
        for folder in folders:
            opt_path = folder / 'model_opt.pt'
            if not opt_path.exists():
                print(f"  {folder.name}: model_opt.pt not found, skipping")
                continue

            opt_data = torch.load(opt_path, map_location='cpu', weights_only=False)
            loss_data = opt_data['loss']
            if isinstance(loss_data, dict):
                component_arrays = [np.asarray(v, dtype=float) for v in loss_data.values() if v]
                if not component_arrays:
                    print(f"  {folder.name}: empty loss history, skipping")
                    continue
                total_loss = np.sum(component_arrays, axis=0)
            else:
                total_loss = np.asarray(loss_data, dtype=float)
            min_loss = float(np.nanmin(total_loss))

            is_nan = np.isnan(min_loss)
            is_inf = np.isinf(min_loss)
            is_near_zero = abs(min_loss) < NEAR_ZERO_THRESHOLD

            if is_nan or is_inf or is_near_zero:
                reason = (
                    "NaN" if is_nan else
                    "inf" if is_inf else
                    f"near-zero ({min_loss:.2e})"
                )
                print(f"  {folder.name}: min loss = {min_loss:.6e}  [SKIPPED: {reason}]")
                continue

            model_losses.append((folder, min_loss))
            print(f"  {folder.name}: min loss = {min_loss:.6e}")

        if not model_losses:
            print(f"  ** No valid models in this group **")
            continue

        # Find the best model in this group
        losses = [loss for _, loss in model_losses]
        best_idx = int(np.argmin(losses))
        best_folder, best_loss = model_losses[best_idx]

        print(f"\n  -> Best: {best_folder.name}  (loss = {best_loss:.6e})")
        results[prefix] = (best_folder, best_loss)

        # Create scatter plot for this group
        plt.figure(figsize=(6, 8))
        plt.scatter(
            [0] * len(losses), losses,
            s=80, alpha=0.7, edgecolors='black', linewidths=0.5,
        )
        plt.scatter(
            [0], [losses[best_idx]],
            s=120, c='red', marker='*', zorder=5, label='Best',
        )
        plt.ylabel('Minimum Loss')
        plt.title(f'Model Losses — {prefix}  ({len(model_losses)} valid run(s))')
        plt.xticks([])
        plt.xlim(-0.5, 0.5)
        plt.legend()

        fig_dir = models_dir.parent / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f'{prefix}_loss_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved to: {fig_path}")

    return results


if __name__ == '__main__':
    results = find_best_models()

    if not results:
        print("\nNo valid models found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for prefix, (folder, loss) in sorted(results.items()):
        print(f"  {prefix}  ->  {folder.name}  (loss = {loss:.6e})")
