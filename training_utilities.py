import torch
from torch.utils.data import Dataset, DataLoader
import plotting_utilities as pu
import numpy as np
import neural_models as nm
import time
import tempfile
import os
from pathlib import Path

from loss_functions import (
    nannorm,
    nanstd,
    log_nanmean_exp,
    corr_loss,
    cos_loss,
    mse_loss,
    combined_loss,
    calc_loss_dict,
)


def _normalized_mse(pred, target, eps=1e-8):
    """MSE normalized by target variance (fraction of variance unexplained).

    NaN entries in *target* are excluded from both MSE and variance.
    """
    valid = ~torch.isnan(target)
    if not valid.any():
        return pred.new_tensor(0.0)
    p, t = pred[valid], target[valid]
    return (p - t).pow(2).mean() / (t.var() + eps)


def _cosine_loss(pred, target, eps=1e-8):
    """Mean cosine distance, masking out near-zero and NaN target vectors."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    has_nan = torch.isnan(target_flat).any(dim=-1)
    mag = target_flat.norm(dim=-1)
    valid = (mag > eps) & ~has_nan
    if not valid.any():
        return pred.new_tensor(0.0)
    cos_sim = torch.nn.functional.cosine_similarity(
        pred_flat[valid], target_flat[valid], dim=-1,
    )
    return 1.0 - cos_sim.mean()


def odometry_loss(pred_rot, pred_trans, target_rot, target_trans, dt):
    """Combined rate + position loss for ego-motion estimation.

    Rotation uses normalized MSE (MSE / var(target)), giving a unitless
    fraction-of-variance-unexplained in the ~0-1 range.  Translation uses
    cosine distance (direction only, since amplitude is unrecoverable),
    also in the 0-1 range.  Both are computed on rates and on integrated
    positions, and the four terms are averaged.

    Args:
        pred_rot: (batch, T, 3) predicted rotation rate (deg/s).
        pred_trans: (batch, T, 3) predicted translation rate.
        target_rot: (batch, T, 3) target rotation rate (deg/s).
        target_trans: (batch, T, 3) target translation rate.
        dt: time step (seconds).
    """
    loss_rot_rate = _normalized_mse(pred_rot, target_rot)
    loss_trans_rate = _cosine_loss(pred_trans, target_trans)

    pred_rot_pos = torch.cumsum(pred_rot * dt, dim=1)
    target_rot_pos = torch.cumsum(target_rot * dt, dim=1)
    loss_rot_pos = _normalized_mse(pred_rot_pos, target_rot_pos)

    pred_trans_pos = torch.cumsum(pred_trans * dt, dim=1)
    target_trans_pos = torch.cumsum(target_trans * dt, dim=1)
    loss_trans_pos = _cosine_loss(pred_trans_pos, target_trans_pos)

    return (loss_rot_rate + loss_trans_rate + loss_rot_pos + loss_trans_pos) / 4


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

def train_model(model, inputs, target, output_neurons, input_neurons, rng=np.random.default_rng(0), model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None, reload_best_on_loss_jump=True, loss_fn=combined_loss,
                patience=100, alpha_factor=0.5, opt_state_dict=None, loss_history=None, reset_init_every=1, 
                sigma_lambda=0, peak_lambda=0, l1_lambda=0, 
                input_noise_std=0.0, contrast_std=0.0, luminance_std=0.0, random_flip_input=False,
                batch_size=None, train_mask=None, fit_odometry=False):

    assert sigma_lambda >= 0, 'lambdas must be positive'
    assert peak_lambda >= 0, 'lambdas must be positive'

    if train_mask is None:
        train_mask = torch.ones(inputs.shape[:3], dtype=torch.bool, device=inputs.device)

    # --- train the model ---
    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    cached_mappings = {'input_mapping': input_neurons, 'output_mapping': output_neurons, 'filter_state': None}
    last_state = None
    last_filter_state = None

    if opt_state_dict is not None:
        opt.load_state_dict(opt_state_dict)
        # Move optimizer state tensors to model device (state dict may have been loaded to CPU)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model.device)

    if loss_history is None:
        loss_plot = []
    else:
        # Preserve prior CPU loss history for plotting/checkpointing
        loss_plot = list(loss_history)

    # GPU-side loss history; append detached scalars each step
    loss_list = []

    # Track best loss entirely on GPU to avoid syncs; seed from prior CPU history if available
    best_seed = min(loss_plot) if len(loss_plot) > 0 else float('inf')
    best_loss = torch.tensor(best_seed, device=model.device, dtype=model.dtype)

    time_since_best = 0
    last_reload_state = None
    last_reload_opt_state = None
    # Index of last GPU loss converted to CPU for plotting
    last_saved_idx = -1

    # --- set up DataLoader for mini-batching ---
    num_stimuli = inputs.shape[0]
    use_batching = batch_size is not None and batch_size < num_stimuli
    if use_batching:
        dataloader = DataLoader(
            StimulusDataset(inputs, target, train_mask),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        batch_iter = iter(dataloader)

    for step in range(num_steps):
        if (step % reset_init_every) == 0:
            last_state = None
            last_filter_state = None

        # --- mini-batch: draw next batch from DataLoader ---
        if use_batching:
            try:
                inputs_batch, target_batch, train_mask_batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                inputs_batch, target_batch, train_mask_batch = next(batch_iter)
            inputs_batch = inputs_batch.clone()
            batch_init_state = None
            batch_filter_state = None
        else:
            inputs_batch = inputs.clone()
            target_batch = target
            train_mask_batch = train_mask
            batch_init_state = last_state
            batch_filter_state = last_filter_state

        # --- data augmentation (applied only to the batch) ---
        if luminance_std > 0:
            luminance = torch.tensor(rng.standard_normal(1) * luminance_std, device=inputs.device, dtype=inputs.dtype)
            inputs_batch = inputs_batch + luminance

        if contrast_std > 0:
            contrast = torch.tensor(np.exp(rng.standard_normal(1) * contrast_std), device=inputs.device, dtype=inputs.dtype)
            inputs_batch = (inputs_batch - torch.mean(inputs_batch)) * contrast + torch.mean(inputs_batch)

        if input_noise_std > 0:
            noise = torch.tensor(rng.standard_normal(inputs_batch.shape) * input_noise_std, device=inputs.device, dtype=inputs.dtype)
            inputs_batch = inputs_batch + noise

        start = time.time()
        opt.zero_grad()

        # Derive time mask from train_mask (True if any column is valid at that time step)
        time_mask = train_mask_batch.any(dim=-1)  # (batch, time)
        # Find first valid time step for slicing (pre-stim is contiguous at start)
        loss_start_step = 0
        if not time_mask[:, 0].all():
            loss_start_step = time_mask.all(dim=0).to(torch.int).argmax().item()

        if fit_odometry:
            # inputs shape: (batch, time, 2*N) — first N right eye, last N left eye
            N = inputs_batch.shape[2] // 2
            right_batch = inputs_batch[:, :, :N]
            left_batch = inputs_batch[:, :, N:]
            # Columns that are never valid in this batch → structurally OOB for the decoder
            oob = ~train_mask_batch.any(dim=1).any(dim=0)  # (2*N,)
            pred_rot, pred_trans, _, _, pred_flow = model.estimate_ego_motion_from_image(
                right_batch, left_batch, oob_columns=oob)

            target_rot = target_batch['rotation'][:, loss_start_step:]
            target_trans = target_batch['translation'][:, loss_start_step:]
            odom_loss = odometry_loss(
                pred_rot[:, loss_start_step:], pred_trans[:, loss_start_step:],
                target_rot, target_trans, model.dt,
            )

            # Flow loss: use train_mask and NaN to select valid elements
            target_flow = target_batch['optic_flow'][:, loss_start_step:]
            pred_flow_valid = pred_flow[:, loss_start_step:]
            flow_mask = train_mask_batch[:, loss_start_step:].unsqueeze(-1).expand_as(target_flow)
            valid_flow = flow_mask & ~torch.isnan(target_flow)
            if valid_flow.any():
                flow_loss = _normalized_mse(pred_flow_valid[valid_flow], target_flow[valid_flow])
            else:
                flow_loss = pred_flow.new_tensor(0.0)

            loss = odom_loss + flow_loss
        else:
            pred, step_state, cached_mappings = model.predict(inputs_batch, init_state=batch_init_state, init_filter_state=batch_filter_state, input_neurons=cached_mappings['input_mapping'], output_neurons=cached_mappings['output_mapping'])
            # Only carry forward hidden state when using full batch (states match stimuli)
            if not use_batching:
                last_state = step_state
                last_filter_state = cached_mappings.get('filter_state')

            if model.has_decoder:
                pred_dec = pred['decoder'][:, loss_start_step:].clone()
                target_dec = target_batch[:, loss_start_step:].clone()
                invalid = ~train_mask_batch[:, loss_start_step:].unsqueeze(-1).expand_as(pred_dec)
                pred_dec[invalid] = float('nan')
                target_dec[invalid] = float('nan')
                loss = loss_fn(target_dec, pred_dec)
            else:
                pred_loss = {k: v[:, loss_start_step:] for k, v in pred['firing_rate'].items()}
                target_loss = {k: v[:, loss_start_step:] for k, v in target_batch.items()}
                loss = calc_loss_dict(loss_fn, pred_loss, target_loss)

        if sigma_lambda > 0:
            # add a regularizer based on the operator norm of the model
            max_sigma_penalty = torch.relu(model.calc_sigma_max(max_iter=10) - model.max_allowed_sigma)**2
            loss = loss + sigma_lambda * max_sigma_penalty

        if peak_lambda > 0 and not fit_odometry:
            # add a regularizer to penalize high values (logmeanexp)
            peak_val = torch.stack([
                log_nanmean_exp(v)
                for v in pred['firing_rate'].values()
            ]).mean()
            loss = loss + peak_lambda * peak_val

        if l1_lambda > 0:
            # L1 regularizer to sparsify strength weights (encourage smaller magnitudes)
            input_l1 = model.input_strength_log.exp().mean()
            output_l1 = model.output_strength_log.exp().mean()

            loss = loss + l1_lambda * (input_l1 + output_l1) / 2

            if model.has_decoder:
                decoder_l1 = torch.stack([p.abs().mean() for p in model.decoder.conv_layers.parameters()]).mean()
                loss = loss + l1_lambda * decoder_l1

        # Record GPU-side loss scalar without sync (must happen before best-loss check
        # so the current loss is included when syncing to loss_plot for checkpoints)
        loss_is_nan = torch.isnan(loss)
        loss_list.append(loss.detach())

        # Detect if you've found the best loss so far BEFORE updating weights (no sync)
        # NaN < anything is False, so NaN losses won't overwrite best_loss
        if loss < best_loss:
            best_loss = loss.detach()
            time_since_best = 0

            # track and save the best checkpoint for easy rollback
            # Save BEFORE opt.step() so we save the weights that achieved this loss
            if model_save_folder:
                # Sync GPU losses to loss_plot before saving so checkpoint has up-to-date history
                for i in range(last_saved_idx + 1, len(loss_list)):
                    loss_plot.append(loss_list[i].item())
                last_saved_idx = len(loss_list) - 1

                model.save(model_save_folder / 'model.pt')
                # Use atomic write for optimizer state to prevent corruption on network filesystems
                # Ensures all-or-nothing writes and prevents partial/corrupted files
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
                    # Atomically replace target file
                    # Note: Removed os.fsync(dir_fd) to prevent deadlock on network filesystems
                    # when multiple processes sync the same directory simultaneously
                    os.replace(tmp_path, str(opt_path))
                except Exception:
                    # Clean up temp file on error
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception:
                        pass
                    raise
        else:
            time_since_best += 1

        # Skip gradient update if loss is NaN to avoid corrupting weights
        # The patience mechanism will still trigger a reload of the best model
        if loss_is_nan:
            print(f'Warning: NaN loss detected at step {step + 1}, skipping gradient update')
        else:
            # Update weights AFTER checking/saving best model
            loss.backward()

            # Optional gradient clipping (prefer norm if both provided)
            if clip_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                if total_norm > clip_grad_norm:
                    print(f'Gradient clipped at step {step + 1}: norm {total_norm:.2f} > max_norm {clip_grad_norm}')
            elif clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad_value)
            opt.step()

        if reload_best_on_loss_jump and (model_save_folder is not None):
            if (time_since_best > patience) or (loss_is_nan):
                opt_path = model_save_folder / 'model_opt.pt'
                if not opt_path.exists():
                    # No best checkpoint saved yet (e.g. NaN on step 1) — skip reload
                    time_since_best = 0
                    print(f'Warning: NaN loss but no checkpoint to reload (step {step + 1}), continuing.')
                else:
                    # Load the best model state into the existing model
                    checkpoint = torch.load(model_save_folder / 'model.pt', map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'])

                    # Load the best optimizer state and reduce learning rate
                    opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                    opt.load_state_dict(opt_state['optimizer_state_dict'])
                    last_reload_state = {k: v.detach().cpu().clone() for k, v in checkpoint['state_dict'].items()}
                    last_reload_opt_state = opt_state['optimizer_state_dict']

                    # Move optimizer state tensors to model device (load_state_dict keeps them on CPU)
                    for state in opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(model.device)

                    alpha = alpha * alpha_factor
                    for param_group in opt.param_groups:
                        param_group['lr'] = alpha

                    # Reset hidden state to avoid mismatch with reloaded weights
                    last_state = None

                    time_since_best = 0
                    print(' -------------------------------- ')
                    print(f'Reloaded best checkpoint. Learning rate reduced to {alpha:.2e}.')
                    print(' -------------------------------- ')

        if (step % save_every == 0) | (step == num_steps - 1):
            with torch.no_grad():
                # Find the model's new steady state should be close ish to the previous one
                # so only do it every save_every steps
                # model.find_steady_state()  # done post-training via analysis/find_steady_state.py

                # Convert only new GPU losses to CPU floats for plotting/logging
                for i in range(last_saved_idx + 1, len(loss_list)):
                    loss_plot.append(loss_list[i].item())
                last_saved_idx = len(loss_list) - 1

                # Print last CPU loss value only at save cadence
                # pu.plot_readout_target(pred, target, save_folder=model_save_folder)
                pu.plot_loss(loss_plot, save_folder=model_save_folder)

        print(f'step {step + 1} / {num_steps} | dt = {(time.time()-start):.3g} | lr = {alpha:.2e} | path = {model_save_folder}')

    # Ensure final GPU losses are synced to CPU history
    if len(loss_list) > 0 and last_saved_idx < len(loss_list) - 1:
        for i in range(last_saved_idx + 1, len(loss_list)):
            loss_plot.append(loss_list[i].item())

    return model, loss_plot, opt.state_dict(), {
        "best_loss": float(best_loss.item()),
        "alpha": alpha,
        "time_since_best": time_since_best,
        "last_reload_state": last_reload_state,
        "last_reload_opt_state": last_reload_opt_state,
    }



