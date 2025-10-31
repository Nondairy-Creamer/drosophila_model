import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

def plot_readout_target(readout, target, readout_names, save_folder=Path('figures'), eps=1e-8):
    save_folder = convert_and_create_save_folder(save_folder)

    for i in range(readout.shape[0]):
        plt.figure()
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        direction = ['progressive', 'regressive', 'up', 'down']
        readout_plot = readout[i].mean(axis=(1, 2)).detach().cpu().numpy().T.reshape((4, readout.shape[1] // 4)).T
        target_plot = target[i].mean(axis=(1, 2)).detach().cpu().numpy().T.reshape((4, target.shape[1] // 4)).T
        
        # scale target to match prediction for visualization
        # since scale doesnt matter for our loss
        target_plot -= np.mean(target_plot)
        target_plot /= np.std(target_plot)
        target_plot *= np.std(readout_plot)  
        target_plot += np.mean(readout_plot)

        for j in range(readout_plot.shape[1]):
            plt.plot(target_plot[:, j], color=colors[j], label=direction[j])
            plt.plot(readout_plot[:, j], color=colors[j], ls='--', label=None)
        plt.title(readout_names[i])
        plt.ylabel('model response')
        plt.xlabel('stimuli')
        plt.legend()
        plt.savefig(save_folder / ('pred_vs_target_' + readout_names[i] + '.png'))
        plt.close()

        # plot with 0 in frame
        max_plot = np.max(readout_plot)
        plot_fudge = max_plot * 0.1
        for j in range(readout_plot.shape[1]):
            plt.plot(target_plot[:, j], color=colors[j], label=direction[j])
            plt.plot(readout_plot[:, j], color=colors[j], ls='--', label=None)
        plt.title(readout_names[i])
        plt.ylabel('model response')
        plt.xlabel('stimuli')
        plt.ylim(-plot_fudge, max_plot + plot_fudge)
        plt.legend()
        plt.savefig(save_folder / ('pred_vs_target_' + readout_names[i] + '_0baseline.png'))
        plt.close()

def plot_loss(loss, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    plt.figure()
    plt.plot(loss)
    plt.ylabel('negative correlation loss')
    plt.xlabel('epochs')
    plt.savefig(save_folder / 'loss.png')
    plt.close()

def plot_response_over_space(response, x, z, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    for si, s in enumerate(response):
        # just plot the first timepoint
        plt.figure()
        sc = plt.scatter(x, z, c=s)
        plt.colorbar(sc)
        plt.xlabel('location in x, posterior <-> anterior')
        plt.ylabel('location in z, ventral <-> dorsal')
        plt.savefig(save_folder / ('resp_' + str(si) + '.png'))
        plt.close()

def convert_and_create_save_folder(save_folder):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder

def scatter_3d_with_origin_axes(xyz, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]

    # Scatter plot
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.8)

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Axis arrows starting at origin (0,0,0)
    axis_length = max_range * 1.2
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='k', arrow_length_ratio=0.1)

    # Labels at tips
    ax.text(axis_length, 0, 0, 'X', color='k', fontsize=12)
    ax.text(0, axis_length, 0, 'Y', color='k', fontsize=12)
    ax.text(0, 0, axis_length, 'Z', color='k', fontsize=12)

    # Labels + colorbar
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter with Origin Axes")

    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Z value (color)")
    plt.savefig(save_folder / '3d_column_position.png')
    plt.close()

