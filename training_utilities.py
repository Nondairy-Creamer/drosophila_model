import torch
from torch.utils.data import Dataset, DataLoader
import plotting_utilities as pu
import numpy as np
import neural_models as nm
import time
import tempfile
import os
from pathlib import Path

import loss_functions as lf


class StimulusDataset(Dataset):
    """Dataset that pairs inputs with targets (tensor or dict of tensors) and a train mask."""
    def __init__(self, inputs, target, train_mask):
        self.inputs = inputs
        self.target = target
        self.train_mask = train_mask
        self._target_is_dict = isinstance(target, dict)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self._target_is_dict:
            return self.inputs[idx], {k: v[idx] for k, v in self.target.items()}, self.train_mask[idx]
        return self.inputs[idx], self.target[idx], self.train_mask[idx]


def train_model(model, inputs, target, output_neurons, input_neurons,
                loss_components,
                rng=np.random.default_rng(0), model_save_folder=None,
                num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None,
                reload_best_on_loss_jump=True,
                patience=10, alpha_factor=0.5,
                opt_state_dict=None, loss_history=None,
                input_noise_std=0.0, contrast_std=0.0, luminance_std=0.0,
                batch_size=None, train_mask=None):

    uses_two_eyes = 'odometry_loss' in loss_components

    if train_mask is None:
        train_mask = torch.ones(inputs.shape[:3], dtype=torch.bool, device=inputs.device)

    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    cached_mappings = {'input_mapping': input_neurons, 'output_mapping': output_neurons}

    if opt_state_dict is not None:
        opt.load_state_dict(opt_state_dict)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model.device)

    loss_plot = list(loss_history) if loss_history else []

    best_loss = torch.tensor(float('inf'), device=model.device, dtype=model.dtype)
    time_since_best = 0
    last_reload_state = None
    last_reload_opt_state = None

    num_stimuli = inputs.shape[0]
    if batch_size is None or batch_size > num_stimuli:
        batch_size = num_stimuli
    dataloader = DataLoader(
        StimulusDataset(inputs, target, train_mask),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    step = 0
    epoch = 0
    while step < num_steps:
        epoch_start = time.time()
        epoch_losses = []

        for inputs_batch, target_batch, train_mask_batch in dataloader:
            if step >= num_steps:
                break

            inputs_batch = inputs_batch.clone()

            if luminance_std > 0:
                luminance = torch.tensor(rng.standard_normal(1) * luminance_std, device=inputs.device, dtype=inputs.dtype)
                inputs_batch = inputs_batch + luminance

            if contrast_std > 0:
                contrast = torch.tensor(np.exp(rng.standard_normal(1) * contrast_std), device=inputs.device, dtype=inputs.dtype)
                inputs_batch = (inputs_batch - torch.mean(inputs_batch)) * contrast + torch.mean(inputs_batch)

            if input_noise_std > 0:
                noise = torch.tensor(rng.standard_normal(inputs_batch.shape) * input_noise_std, device=inputs.device, dtype=inputs.dtype)
                inputs_batch = inputs_batch + noise

            opt.zero_grad()

            time_mask = train_mask_batch.any(dim=-1)
            loss_start_step = 0
            if not time_mask[:, 0].all():
                loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()

            if uses_two_eyes:
                N = inputs_batch.shape[2] // 2
                right_batch = inputs_batch[:, :, :N]
                left_batch = inputs_batch[:, :, N:]
                oob = ~train_mask_batch.any(dim=1).any(dim=0)
                pred_rot, pred_trans, right_res, left_res = \
                    model.estimate_ego_motion_from_image(
                        right_batch, left_batch, oob_columns=oob)

                outputs = {}
                for key in list(right_res['outputs'].keys()):
                    if key in ('voltage', 'firing_rate'):
                        continue
                    outputs[key] = torch.cat(
                        [right_res['outputs'][key],
                         left_res['outputs'][key]], dim=2)
                outputs['pred_rotation'] = pred_rot
                outputs['pred_translation'] = pred_trans
            else:
                result = model.predict(
                    inputs_batch,
                    input_neurons=cached_mappings['input_mapping'],
                    output_neurons=cached_mappings['output_mapping'])
                outputs = result['outputs']
                cached_mappings = result['mappings']

            loss = sum(
                getattr(lf, name)(model, outputs, target_batch,
                                  train_mask_batch, loss_start_step)
                for name in loss_components
            ) / len(loss_components)

            if torch.isnan(loss):
                print(f'Warning: NaN loss at step {step + 1}, skipping gradient update')
            else:
                loss.backward()
                if clip_grad_norm is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    if total_norm > clip_grad_norm:
                        print(f'Gradient clipped at step {step + 1}: norm {total_norm:.2f} > max_norm {clip_grad_norm}')
                elif clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
                opt.step()
                epoch_losses.append(loss.detach())

            step += 1

        # --- End of epoch ---
        epoch += 1

        if epoch_losses:
            epoch_avg = torch.stack(epoch_losses).mean()
            loss_plot.append(epoch_avg.item())
            if epoch_avg < best_loss:
                best_loss = epoch_avg.detach()
                time_since_best = 0
                if model_save_folder:
                    model.save(model_save_folder / 'model.pt')
                    opt_path = Path(model_save_folder / 'model_opt.pt')
                    opt_payload = {
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': loss_plot,
                        'step': step
                    }
                    try:
                        with tempfile.NamedTemporaryFile('wb', delete=False, dir=str(opt_path.parent), suffix='.tmp') as tmp:
                            tmp_path = tmp.name
                            torch.save(opt_payload, tmp_path)
                            tmp.flush()
                            os.fsync(tmp.fileno())
                        os.replace(tmp_path, str(opt_path))
                    except Exception:
                        try:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        except Exception:
                            pass
                        raise
            else:
                time_since_best += 1
        else:
            time_since_best += 1

        if reload_best_on_loss_jump and (model_save_folder is not None):
            if time_since_best > patience:
                opt_path = model_save_folder / 'model_opt.pt'
                if not opt_path.exists():
                    time_since_best = 0
                    print(f'Warning: no checkpoint to reload (epoch {epoch}), continuing.')
                else:
                    checkpoint = torch.load(model_save_folder / 'model.pt', map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'])

                    opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                    opt.load_state_dict(opt_state['optimizer_state_dict'])
                    last_reload_state = {k: v.detach().cpu().clone() for k, v in checkpoint['state_dict'].items()}
                    last_reload_opt_state = opt_state['optimizer_state_dict']

                    for state in opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(model.device)

                    alpha = alpha * alpha_factor
                    patience = int(patience * np.sqrt(1 / alpha_factor))
                    for param_group in opt.param_groups:
                        param_group['lr'] = alpha

                    time_since_best = 0
                    best_loss = torch.tensor(float('inf'), device=model.device, dtype=model.dtype)
                    print(' -------------------------------- ')
                    print(f'Reloaded best checkpoint. Learning rate reduced to {alpha:.2e}.')
                    print(' -------------------------------- ')

        if (epoch % save_every == 0) or (step >= num_steps):
            pu.plot_loss(loss_plot, save_folder=model_save_folder)

        epoch_loss_str = f'{epoch_avg.item():.4f}' if epoch_losses else 'NaN'
        print(f'epoch {epoch} | steps {step}/{num_steps} | loss = {epoch_loss_str} | dt = {time.time()-epoch_start:.3g}s | lr = {alpha:.2e} | path = {model_save_folder}')

    return model, loss_plot, opt.state_dict(), {
        "best_loss": float(best_loss.item()),
        "alpha": alpha,
        "time_since_best": time_since_best,
        "last_reload_state": last_reload_state,
        "last_reload_opt_state": last_reload_opt_state,
    }
