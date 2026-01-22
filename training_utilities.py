import torch
import plotting_utilities as pu
import numpy as np
import neural_models as nm
import time
import tempfile
import os
from pathlib import Path

def nannorm(unormalized_tensor, keepdim=False, dim=None, eps=1e-6):
    return torch.nansum(torch.abs(unormalized_tensor)**2, dim=dim, keepdim=keepdim).sqrt().clamp_min(eps)

def nanstd(x, dim, keepdim=False, eps=1e-6):
    mask = ~torch.isnan(x)
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    x0 = torch.where(mask, x, 0.0)
    mean = x0.sum(dim=dim, keepdim=True) / count
    var = torch.where(mask, (x - mean)**2, 0.0).sum(dim=dim, keepdim=True) / count
    std = var.sqrt().clamp_min(eps)
    if keepdim:
        return std
    if isinstance(dim, tuple):
        for d in sorted(dim, reverse=True):
            std = std.squeeze(d)
        return std
    return std.squeeze(dim)

def log_nanmean_exp(x, dim=None):
    """Log of mean of exp, ignoring NaNs. Numerically stable."""
    if dim is None:
        x = x.flatten()
        dim = 0
    # Replace NaNs with -inf so exp(-inf) = 0
    x_clean = torch.where(torch.isnan(x), torch.full_like(x, -float('inf')), x)
    # Count non-NaN elements
    n = (~torch.isnan(x)).sum(dim=dim).clamp_min(1)
    return torch.logsumexp(x_clean, dim=dim) - torch.log(n.float())

def corr_loss(y_hat, y, dim=None, eps=1e-6):
    # Create mask of positions valid in BOTH tensors
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y)).float()
    valid_count = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1)

    # Replace NaN with 0 to prevent NaN gradient propagation (mask handles statistics)
    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

    # Compute means over valid positions only
    y_hat_mean = (y_hat * valid_mask).sum(dim=dim, keepdim=True) / valid_count
    y_mean = (y * valid_mask).sum(dim=dim, keepdim=True) / valid_count

    # Center the data (mask zeros out invalid positions)
    y_hat_centered = (y_hat - y_hat_mean) * valid_mask
    y_centered = (y - y_mean) * valid_mask

    # Compute std over valid positions
    y_hat_std = (y_hat_centered.pow(2).sum(dim=dim, keepdim=True) / valid_count).sqrt().clamp_min(eps)
    y_std = (y_centered.pow(2).sum(dim=dim, keepdim=True) / valid_count).sqrt().clamp_min(eps)

    # Normalize
    y_hat_norm = y_hat_centered / y_hat_std
    y_norm = y_centered / y_std

    # Compute correlation over valid positions
    corr = (y_hat_norm * y_norm).sum(dim=dim, keepdim=True) / valid_count

    ave_corr = torch.mean(corr)

    return 1.0 - ave_corr

def cos_loss(y_hat, y, dim=None, eps=1e-6):
    # Create mask of positions valid in BOTH tensors
    valid_mask = (~torch.isnan(y_hat) & ~torch.isnan(y)).float()

    # Replace NaN with 0 to prevent NaN gradient propagation (mask handles statistics)
    y_hat = torch.where(torch.isnan(y_hat), torch.zeros_like(y_hat), y_hat)
    y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

    # Apply mask so norms are computed only over valid positions
    y_hat_masked = y_hat * valid_mask
    y_masked = y * valid_mask

    # Normalize (nannorm uses nansum internally, but all values are now valid)
    y_hat_norm = y_hat_masked / nannorm(y_hat_masked, dim=dim, keepdim=True).clamp_min(eps)
    y_norm = y_masked / nannorm(y_masked, dim=dim, keepdim=True).clamp_min(eps)

    # Cosine similarity over valid positions
    cosine_sim = torch.sum(y_hat_norm * y_norm, dim=dim, keepdim=True)

    ave_cosine_sim = torch.mean(cosine_sim)

    return 1.0 - ave_cosine_sim

def combined_loss(y_hat, y, eps=1e-6):
    cos_loss_val = cos_loss(y_hat, y, eps=eps)
    corr_loss_val = corr_loss(y_hat, y, eps=eps)
    return 0.5 * cos_loss_val + 0.5 * corr_loss_val

def calc_loss(loss_fn, pred, target):
    keys = list(pred.keys())
    num_readouts = len(keys)

    loss = loss_fn(pred[keys[0]], target[keys[0]]) / num_readouts
    for i in keys[1:]:
        loss = loss + loss_fn(pred[i], target[i]) / num_readouts

    return loss

def train_model(model, inputs, target, output_neurons, input_neurons, model_save_folder=None, num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None, reload_best_on_loss_jump=True, loss_fn=combined_loss,
                patience=100, alpha_factor=0.5, opt_state_dict=None, loss_history=None, reset_init_every=1, 
                sigma_lambda=0, peak_lambda=0, l1_lambda=0):

    assert sigma_lambda >= 0, 'lambdas must be positive'
    assert peak_lambda >= 0, 'lambdas must be positive'

    # --- train the model ---
    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    cached_mappings = {'input_mapping': input_neurons, 'output_mapping': output_neurons}
    last_state = None

    if opt_state_dict is not None:
        opt.load_state_dict(opt_state_dict)

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
    # Index of last GPU loss converted to CPU for plotting
    last_saved_idx = -1

    for step in range(num_steps):
        if (step % reset_init_every) == 0:
            last_state = None

        start = time.time()
        opt.zero_grad()
        pred, last_state, cached_mappings = model.predict(inputs, init_state=last_state, input_neurons=cached_mappings['input_mapping'], output_neurons=cached_mappings['output_mapping'])

        loss = calc_loss(loss_fn, pred['firing_rate'], target)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
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
                model.find_steady_state()

                # Convert only new GPU losses to CPU floats for plotting/logging
                for i in range(last_saved_idx + 1, len(loss_list)):
                    loss_plot.append(loss_list[i].item())
                last_saved_idx = len(loss_list) - 1

                # Print last CPU loss value only at save cadence
                # pu.plot_readout_target(pred, target, save_folder=model_save_folder)
                pu.plot_loss(loss_plot, save_folder=model_save_folder)

        print(f'step {step + 1} / {num_steps} | dt = {(time.time()-start):.3g} | lr = {alpha:.2e} | path = {model_save_folder}')




