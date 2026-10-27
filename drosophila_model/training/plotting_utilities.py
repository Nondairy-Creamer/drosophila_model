"""Training-time plots (the loss curve drawn by ``fit``)."""
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


def plot_loss(component_losses, save_folder):
    """Plot total loss and per-component losses side by side.

    component_losses: dict mapping component name -> list of per-epoch values.
    Saves ``<save_folder>/loss.png``.
    """
    save_folder = convert_and_create_save_folder(save_folder)

    components = {k: np.asarray(v, dtype=float) for k, v in component_losses.items() if v}
    if not components:
        return

    total = np.sum(list(components.values()), axis=0)
    # A one-epoch history is a single point, which a plain line does not draw.
    marker = 'o' if len(total) == 1 else None

    has_multiple = len(components) > 1
    if has_multiple:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    ax1.plot(total, marker=marker)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax1.set_title('Total loss')
    finite_mask = np.isfinite(total)
    if finite_mask.any():
        ax1.set_ylim(0.9 * np.nanmin(total[finite_mask]),
                     1.1 * np.nanpercentile(total[finite_mask], 90))

    if ax2 is not None:
        for name, arr in components.items():
            ax2.plot(arr, label=name, marker=marker)
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epochs')
        ax2.set_title('Loss components')
        ax2.legend()
        all_vals = np.concatenate(list(components.values()))
        finite = np.isfinite(all_vals)
        if finite.any():
            ax2.set_ylim(0.9 * np.nanmin(all_vals[finite]),
                         1.1 * np.nanpercentile(all_vals[finite], 90))

    fig.tight_layout()
    fig.savefig(save_folder / 'loss.png')
    plt.close(fig)


def convert_and_create_save_folder(save_folder):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder
