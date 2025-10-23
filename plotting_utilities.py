import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path

def plot_readout_target(readout, target, readout_names, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    for i in range(readout.shape[2]):
        plt.figure()
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        direction = ['progressive', 'regressive', 'up', 'down']
        readout_plot = readout[:, :, i].mean(axis=1).detach().cpu().numpy().T.reshape((4, readout.shape[0] // 4)).T
        target_plot = target[:, :, i].mean(axis=1).detach().cpu().numpy().T.reshape((4, target.shape[0] // 4)).T
        for j in range(readout_plot.shape[1]):
            plt.plot(target_plot[:, j], color=colors[j], label=direction[j])
            plt.plot(readout_plot[:, j], color=colors[j], ls='--', label=None)
        plt.title(readout_names[i])
        plt.ylabel('model response')
        plt.xlabel('stimuli')
        plt.legend()
        plt.savefig(save_folder / ('pred_vs_target_' + readout_names[i] + '.png'))
        plt.close()

def plot_loss(loss, save_folder=Path('figures')):
    save_folder = convert_and_create_save_folder(save_folder)

    plt.figure()
    plt.plot(loss)
    plt.ylabel('MSE loss')
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