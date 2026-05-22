import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

board_dim = [40, 30]  # inches x by y
drone_dist = 12  # inches
wavelength_angle = 45
wavelength_inches = drone_dist * np.tan(wavelength_angle * np.pi / 180)

samples_per_inch = 10
nx = int(board_dim[0] * samples_per_inch)
ny = int(board_dim[1] * samples_per_inch)

figures_dir = os.path.join("figures", "board_stimulus")
os.makedirs(figures_dir, exist_ok=True)


def save_stimulus(stim, path):
    fig, ax = plt.subplots(figsize=(board_dim[0], board_dim[1]), dpi=samples_per_inch)
    ax.imshow(stim, cmap="gray", vmin=-1, vmax=1, origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=samples_per_inch, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


x = np.linspace(0, board_dim[0], nx, endpoint=False)
y = np.linspace(0, board_dim[1], ny, endpoint=False)
xx, yy = np.meshgrid(x, y)
sine_stim = np.sin(2 * np.pi * xx / wavelength_inches) * np.sin(2 * np.pi * yy / wavelength_inches)
save_stimulus(sine_stim, os.path.join(figures_dir, "sine_stimulus.png"))


rng = np.random.default_rng(1)
noise = rng.standard_normal((ny, nx))
target_freq = np.sqrt(2) / wavelength_inches  # peak of the sine's Fourier energy (cycles/inch)
sigma_ratio = 1.6  # sigma2 / sigma1; ~Marr's LoG approximation
sigma1_inches = np.sqrt(np.log(sigma_ratio**2) / (2 * np.pi**2 * (sigma_ratio**2 - 1))) / target_freq
sigma2_inches = sigma_ratio * sigma1_inches
sigma1_samples = sigma1_inches * samples_per_inch
sigma2_samples = sigma2_inches * samples_per_inch
dog = gaussian_filter(noise, sigma=sigma1_samples) - gaussian_filter(noise, sigma=sigma2_samples)
noise_stim = dog / np.max(np.abs(dog))
save_stimulus(noise_stim, os.path.join(figures_dir, "white_noise_stimulus.png"))
