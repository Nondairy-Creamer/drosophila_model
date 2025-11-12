import torch
from pathlib import Path
import neural_models as nm
import data_utilities as du
import training_utilities as tu

model_folder = Path('models/20251112_172753_good_mapping')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model.pt', freeze_params=False)
stimulus = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
# target is now a dict of tensors; load and move entries to the right device/dtype
_target_loaded = torch.load(model_folder / 'target.pt')
target = {k: (v.to(device=model.device, dtype=model.dtype) if torch.is_tensor(v) else v)
            for k, v in _target_loaded.items()}
num_steps = int(1e6)
alpha = None

input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
loss_fn = getattr(tu, config['training']['loss_fn'])
loss_scaled = tu.get_scaled_loss_fn(loss_fn, contrast_scaling_factor)

# Load the best optimizer state and reduce learning rate
opt_state = torch.load(model_folder / 'model_opt.pt', map_location='cpu', weights_only=False)
# if you want to use the previous alpha, set the alpha here to None
# otherwise, overwrite the previous alpha
if alpha is None:
    alpha = opt_state['optimizer_state_dict']['param_groups'][0]['lr']
else:
    for param_group in opt_state['optimizer_state_dict']['param_groups']:
        param_group['lr'] = alpha

# --- train the model ---
tu.train_model(model, stimulus, target, output_neurons, input_neurons,
    model_save_folder=model_folder,
    num_steps=config['training']['num_steps'],
    save_every=config['training']['save_every'], 
    alpha=config['training']['alpha'], 
    clip_grad_norm=config['training']['clip_norm'],
    loss_fn=loss_scaled)

               