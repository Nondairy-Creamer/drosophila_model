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


class StimulusDataset(Dataset):
    """Dataset that pairs inputs with targets (tensor or dict of tensors)."""
    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target
        self._target_is_dict = isinstance(target, dict)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self._target_is_dict:
            return self.inputs[idx], {k: v[idx] for k, v in self.target.items()}
        return self.inputs[idx], self.target[idx]

def train_model(model, inputs, target, output_neurons, input_neurons, rng=np.random.default_rng(0), model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None, reload_best_on_loss_jump=True, loss_fn=combined_loss,
                patience=100, alpha_factor=0.5, opt_state_dict=None, loss_history=None, reset_init_every=1, 
                sigma_lambda=0, peak_lambda=0, l1_lambda=0, 
                input_noise_std=0.0, contrast_std=0.0, luminance_std=0.0, random_flip_input=False,
                batch_size=None, pre_stim_time=0.0):

    assert sigma_lambda >= 0, 'lambdas must be positive'
    assert peak_lambda >= 0, 'lambdas must be positive'

    # Number of initial time steps to exclude from loss (pre-stimulus period)
    loss_start_step = int(round(pre_stim_time / model.dt)) if pre_stim_time > 0 else 0

    # --- train the model ---
    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    cached_mappings = {'input_mapping': input_neurons, 'output_mapping': output_neurons}
    last_state = None

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
            StimulusDataset(inputs, target),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        batch_iter = iter(dataloader)

    for step in range(num_steps):
        if (step % reset_init_every) == 0:
            last_state = None

        # --- mini-batch: draw next batch from DataLoader ---
        if use_batching:
            try:
                inputs_batch, target_batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                inputs_batch, target_batch = next(batch_iter)
            inputs_batch = inputs_batch.clone()
            batch_init_state = None
        else:
            inputs_batch = inputs.clone()
            target_batch = target
            batch_init_state = last_state

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
        pred, step_state, cached_mappings = model.predict(inputs_batch, init_state=batch_init_state, input_neurons=cached_mappings['input_mapping'], output_neurons=cached_mappings['output_mapping'])
        # Only carry forward hidden state when using full batch (states match stimuli)
        if not use_batching:
            last_state = step_state

        if model.use_decoder:
            # When using decoder, pred['decoder'] and target are tensors, not dicts
            # decoder output: (batches, time, num_columns, 2)
            # target (optic_flow): (batches, time, num_columns, 2)
            pred_loss = pred['decoder'][:, loss_start_step:]
            target_loss = target_batch[:, loss_start_step:]
            loss = loss_fn(target_loss, pred_loss)
        else:
            pred_loss = {k: v[:, loss_start_step:] for k, v in pred['firing_rate'].items()}
            target_loss = {k: v[:, loss_start_step:] for k, v in target_batch.items()}
            loss = calc_loss_dict(loss_fn, pred_loss, target_loss)

        if sigma_lambda > 0:
            # add a regularizer based on the operator norm of the model
            max_sigma_penalty = torch.relu(model.calc_sigma_max(max_iter=10) - model.max_allowed_sigma)**2
            loss = loss + sigma_lambda * max_sigma_penalty

        if peak_lambda > 0:
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

            if model.use_decoder:
                conv1_l1 = model.decoder_conv_1.abs().mean()
                conv2_l1 = model.decoder_conv_2.abs().mean()

                loss = loss + l1_lambda * (conv1_l1 + conv2_l1) / 2

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
                # Load the best model state into the existing model
                checkpoint = torch.load(model_save_folder / 'model.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])

                # Load the best optimizer state and reduce learning rate
                opt_state = torch.load(model_save_folder / 'model_opt.pt', map_location='cpu', weights_only=False)
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



