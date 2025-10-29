import torch
from pathlib import Path
import neural_models as nm
import data_utilities as du
import training_utilities as tu

model_folder = Path('models/20251029_204432')

config = du.load_config(model_folder / 'config.toml')
best_ckpt = model_folder / 'best.pt'
model = nm.NeuralRnn.load(best_ckpt if best_ckpt.exists() else (model_folder / 'model.pt'))
cell_inputs = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
target = torch.load(model_folder / 'target.pt').to(device=model.device)

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
def loss_scaled(y_hat, y):
    return tu.combined_loss(y_hat, y, neuron_scaling=contrast_scaling_factor, eps=1e-6)

alpha = 1e-2  # lower LR when resuming from best
num_steps = int(1e6)

tu.train_model(model, cell_inputs, target, model_save_folder=model_folder, 
               num_steps=num_steps, save_every=config['training']['save_every'], 
               clip_grad_norm=config['training']['clip_norm'], alpha=alpha, loss_fn=loss_scaled)


               