import torch
from pathlib import Path
import argparse
import shutil
import neural_models as nm
import data_utilities as du
import training_utilities as tu

parser = argparse.ArgumentParser(description='Continue training a model from checkpoint')
parser.add_argument('--model-folder', default='models/20260114_205655_gpu9_run1',
                    help='Path to model folder')
parser.add_argument('--new-session', action='store_true', default=False,
                    help='Start a new training session (do not carry over loss history)')
args = parser.parse_args()

model_folder = Path(args.model_folder)
new_session = args.new_session

# If new session, copy folder to <folder_name>_cont and use that
if new_session:
    new_folder = model_folder.parent / f"{model_folder.name}_cont"
    print(f"New session: copying {model_folder} to {new_folder}")
    shutil.copytree(model_folder, new_folder, dirs_exist_ok=True)
    model_folder = new_folder

local_config = du.load_config()
config = du.load_config(model_folder / 'config.toml')
model = nm.BaseRnn.load(model_folder / 'model.pt', freeze_params=False)
_stim_loaded = torch.load(model_folder / 'cell_inputs.pt')
if isinstance(_stim_loaded, dict):
    stimulus = _stim_loaded['cell_inputs'].to(device=model.device)
    train_mask = _stim_loaded['train_mask'].to(device=model.device)
else:
    stimulus = _stim_loaded.to(device=model.device)
    train_mask = None
# target is now a dict of tensors; load and move entries to the right device/dtype
_target_loaded = torch.load(model_folder / 'target.pt')
target = {k: (v.to(device=model.device, dtype=model.dtype) if torch.is_tensor(v) else v)
            for k, v in _target_loaded.items()}
num_steps = int(1e6)
alpha = local_config['training']['alpha'] if new_session else None

input_neurons = ['L1', 'L2', 'L3']  # Convert list to tuple for compatibility
output_neurons = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

# Create loss function
loss_fn = getattr(tu, config['training']['loss_fn'])

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
loss_history = None if new_session else opt_state['loss']
opt_state_dict = None if new_session else opt_state['optimizer_state_dict']

tu.train_model(model, stimulus, target, output_neurons, input_neurons,
    model_save_folder=model_folder,
    num_steps=num_steps,
    save_every=config['training']['save_every'], 
    alpha=alpha,
    clip_grad_norm=config['training']['clip_norm'],
    loss_fn=loss_fn,
    opt_state_dict=opt_state_dict,
    loss_history=loss_history,
    sigma_lambda=config['training']['sigma_lambda'],
    peak_lambda=config['training']['peak_lambda'],    
    train_mask=train_mask,
    )

               