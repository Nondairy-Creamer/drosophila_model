import torch
from pathlib import Path
import neural_models as nm
import data_utilities as du
import training_utilities as tu

model_folder = Path('models/20251104_203752')

config = du.load_config(model_folder / 'config.toml')
model = nm.NeuralRnn.load(model_folder / 'model.pt')
cell_inputs = torch.load(model_folder / 'cell_inputs.pt').to(device=model.device)
target = torch.load(model_folder / 'target.pt').to(device=model.device)
num_steps = int(1e6)
alpha = None

# Load contrast scaling factor and create loss function
contrast_scaling_factor_path = model_folder / 'contrast_scaling_factor.pt'
contrast_scaling_factor = torch.load(contrast_scaling_factor_path).to(device=model.device)
loss_fn = getattr(tu, config['training']['loss_fn'])
loss_scaled = tu.get_scaled_loss(loss_fn, contrast_scaling_factor)

# Load the best optimizer state and reduce learning rate
opt_state = torch.load(model_folder / 'model_opt.pt', map_location='cpu', weights_only=False)
# if you want to use the previous alpha, set the alpha here to None
# otherwise, overwrite the previous alpha
if alpha is None:
    alpha = opt_state['optimizer_state_dict']['param_groups'][0]['lr']
else:
    for param_group in opt_state['optimizer_state_dict']['param_groups']:
        param_group['lr'] = alpha
        
tu.train_model(model, cell_inputs, target, model_save_folder=model_folder, 
               num_steps=num_steps, save_every=config['training']['save_every'], 
               clip_grad_norm=config['training']['clip_norm'], alpha=alpha, loss_fn=loss_scaled,
               opt_state_dict=opt_state['optimizer_state_dict'], loss_history=opt_state['loss'])


               