import torch
from torch.utils.data import Dataset, DataLoader
from drosophila_model.training import plotting_utilities as pu
import numpy as np
from drosophila_model.model_definition import neural_models as nm
import time
import tempfile
import os
import subprocess
from pathlib import Path

from drosophila_model.training import loss_functions as lf


def get_git_commit_hash():
    """Get the current git commit hash, or None if not in a git repo or git unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def load_processed_stimulus(config, device=None, dtype=None, train_fraction=None):
    """Load pre-baked stimulus dataset(s) for training.

    Reads ``config['stimulus']['stimulus_type']`` (a string or list of
    strings), loads each training file from
    ``<data_folder>/stimulus/training/<type>.pt`` (produced by
    ``create_training_data.py``), validates that all share the same eye
    count and camera offset, and concatenates them along the example axis.
    Stimuli are no longer generated on the fly during training.

    Only the first ``train_fraction`` of each dataset's examples is kept; the
    last ``(1 - train_fraction)`` is cropped off and reserved for testing.
    When ``train_fraction`` is ``None`` it is read from the data config's
    ``[shared].train_fraction`` (see ``config_default_data.toml``).

    Returns a dict with keys ``stimulus``, ``optic_flow``, ``train_mask``,
    ``rotation``, ``translation``, ``num_eyes``, ``eye_offset``.
    ``rotation``/``translation`` are ``None`` for one-eye datasets.
    """
    if device is None:
        device = torch.device(config['model']['device'])
    if dtype is None:
        dtype = getattr(torch, config['model']['dtype'])
    if train_fraction is None:
        train_fraction = config['data']['shared']['train_fraction']
    if not 0 < train_fraction <= 1:
        raise ValueError(
            f"train_fraction must be in (0, 1], got {train_fraction}."
        )

    training_dir = Path(config['data']['data_folder']) / 'stimulus' / 'training'

    stimulus_type = config['stimulus']['stimulus_type']
    stimulus_types = [stimulus_type] if isinstance(stimulus_type, str) else list(stimulus_type)

    stim_list, flow_list, mask_list, rot_list, trans_list = [], [], [], [], []
    num_eyes_set, eye_offsets = set(), set()
    for stim_type in stimulus_types:
        training_path = training_dir / f'{stim_type}.pt'
        if not training_path.exists():
            raise FileNotFoundError(
                f"Training stimulus not found: {training_path}. Generate it first with: "
                f"python drosophila_model/data_handling/create_training_data.py {stim_type}"
            )
        data = torch.load(training_path, map_location='cpu', weights_only=False)
        # Keep the first train_fraction of examples (dim 0); the remainder is
        # held out for testing. Always keep at least one example.
        num_examples = data['stimulus'].shape[0]
        num_train = max(1, int(np.round(num_examples * train_fraction)))
        num_eyes_set.add(data['num_eyes'])
        eye_offsets.add(tuple(data['eye_offset']) if data['eye_offset'] is not None else None)
        stim_list.append(data['stimulus'][:num_train].to(device=device, dtype=dtype))
        flow_list.append(data['optic_flow'][:num_train].to(device=device, dtype=dtype))
        mask_list.append(data['train_mask'][:num_train].to(device=device))
        if data['rotation'] is not None:
            rot_list.append(data['rotation'][:num_train].to(device=device, dtype=dtype))
            trans_list.append(data['translation'][:num_train].to(device=device, dtype=dtype))

    if len(num_eyes_set) > 1:
        raise ValueError(
            f"Mixed eye counts across training stimulus types {stimulus_types}: "
            f"{sorted(num_eyes_set)}. Train one eye geometry at a time."
        )
    if len(eye_offsets) > 1:
        raise ValueError(
            f"Mixed camera offsets across training stimulus types {stimulus_types}: "
            f"{sorted(o for o in eye_offsets if o is not None)}. "
            "Train one rig geometry at a time."
        )

    return {
        'stimulus': torch.cat(stim_list, dim=0),
        'optic_flow': torch.cat(flow_list, dim=0),
        'train_mask': torch.cat(mask_list, dim=0),
        'rotation': torch.cat(rot_list, dim=0) if rot_list else None,
        'translation': torch.cat(trans_list, dim=0) if trans_list else None,
        'num_eyes': num_eyes_set.pop(),
        'eye_offset': eye_offsets.pop(),
    }


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
                rng=None, model_save_folder=None,
                num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, clip_grad_value=None,
                reload_best_on_loss_jump=True,
                patience=10, alpha_factor=0.5,
                opt_state_dict=None, loss_history=None,
                input_noise_std=0.0, contrast_std=0.0, luminance_std=0.0,
                batch_size=None, train_mask=None, eye_offset=None):

    if rng is None:
        rng = np.random.default_rng(0)

    uses_two_eyes = 'odometry_loss' in loss_components

    # Accept the string 'none' (case-insensitive) from TOML configs as a
    # disable sentinel for the clipping knobs, since TOML has no real null.
    if isinstance(clip_grad_norm, str) and clip_grad_norm.lower() == 'none':
        clip_grad_norm = None
    if isinstance(clip_grad_value, str) and clip_grad_value.lower() == 'none':
        clip_grad_value = None

    if train_mask is None:
        train_mask = torch.ones(inputs.shape[:3], dtype=torch.bool, device=inputs.device)

    opt = torch.optim.Adam(model.parameters(), lr=alpha)
    
    # Region-agnostic mapping round-trip. The first iteration passes the
    # caller-provided cell-type-name lists; predict() rebuilds the per-region
    # stim<->cell mappings and we reuse those on every later iteration so the
    # index work happens only once, regardless of which regions exist.
    predict_input_neurons = input_neurons
    predict_output_neurons = output_neurons

    if opt_state_dict is not None:
        opt.load_state_dict(opt_state_dict)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(model.device)

    reg_names = [name for name, mod in model._readout_modules.items()
                 if getattr(mod, 'training_only', False)]
    all_component_names = list(loss_components) + reg_names

    if loss_history:
        loss_plot = {name: list(loss_history.get(name, [])) for name in all_component_names}
    else:
        loss_plot = {name: [] for name in all_component_names}

    best_loss = torch.tensor(float('inf'), device=model.device, dtype=model.dtype)
    time_since_best = 0
    last_reload_state = None
    last_reload_opt_state = None

    num_stimuli = inputs.shape[0]
    if batch_size is None or batch_size > num_stimuli:
        batch_size = num_stimuli
    # Seed the DataLoader's shuffle from the same numpy generator so batch
    # order is reproducible across runs with the same seed (otherwise shuffle
    # draws from the un-seeded global torch RNG).
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(rng.integers(0, 2**63 - 1)))
    dataloader = DataLoader(
        StimulusDataset(inputs, target, train_mask),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=shuffle_generator,
    )

    step = 0
    epoch = 0
    while step < num_steps:
        epoch_start = time.time()
        epoch_losses = []
        epoch_component_losses = {name: [] for name in all_component_names}

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
                    model.regions['optic_lobe'].estimate_ego_motion_from_image(
                        right_batch, left_batch, oob_columns=oob,
                        camera_offset=eye_offset)

                # Per-region outputs: concatenate right+left readouts on the
                # column axis for the two-eye loss terms.
                outputs = {}
                right_optic = right_res['outputs'].get('optic_lobe', {})
                left_optic = left_res['outputs'].get('optic_lobe', {})
                for key in right_optic.keys():
                    if key in ('voltage', 'firing_rate', 'synaptic_current'):
                        continue
                    r, l = right_optic[key], left_optic[key]
                    if r.dim() < 2:
                        outputs[key] = r
                        continue
                    if r.dim() == 2:
                        r = r.unsqueeze(-1)
                        l = l.unsqueeze(-1)
                    outputs[key] = torch.cat([r, l], dim=2)
                outputs['pred_rotation'] = pred_rot
                outputs['pred_translation'] = pred_trans
            else:
                result = model.predict(
                    inputs_batch,
                    input_neurons=predict_input_neurons,
                    output_neurons=predict_output_neurons)
                # Flatten every region's readouts into one dict for the loss
                # functions; readout names are unique across regions.
                outputs = {}
                for region_outputs in result['outputs'].values():
                    outputs.update(region_outputs)
                # Round-trip the rebuilt mappings: output mappings are
                # per-region, while the single combined input mapping comes
                # from whichever region actually builds one.
                mappings = result['mappings']
                predict_output_neurons = {
                    name: m['output_mapping'] for name, m in mappings.items()
                }
                predict_input_neurons = next(
                    (m['input_mapping'] for m in mappings.values()
                     if m.get('input_mapping') is not None),
                    None,
                )

            component_vals = {
                name: getattr(lf, name)(model, outputs, target_batch,
                                        train_mask_batch, loss_start_step)
                for name in loss_components
            }
            for name, mod in model._readout_modules.items():
                if getattr(mod, 'training_only', False) and name in outputs:
                    component_vals[name] = outputs[name]
            loss = sum(component_vals.values())

            # Skip gradient update only on non-finite losses: a NaN/Inf
            # backward() would propagate through every weight and the loss
            # would stay non-finite forever. Finite-but-large outlier batches
            # are left in; if they actually harm training, the plateau
            # detection below reloads the best checkpoint and drops the LR.
            if not torch.isfinite(loss):
                print(f'Warning: non-finite loss ({loss.item():.3e}) at step {step + 1}, skipping gradient update')
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
                for name, val in component_vals.items():
                    epoch_component_losses[name].append(val.detach())

            step += 1

        # --- End of epoch ---
        epoch += 1

        if epoch_losses:
            for name in all_component_names:
                vals = epoch_component_losses[name]
                if vals:
                    loss_plot[name].append(torch.stack(vals).mean().item())
            epoch_avg = torch.stack(epoch_losses).mean()
            if epoch_avg < best_loss:
                best_loss = epoch_avg.detach()
                time_since_best = 0
                if model_save_folder:
                    model.save(model_save_folder / 'model.pt')
                    opt_path = Path(model_save_folder / 'model_opt.pt')
                    opt_payload = {
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': {k: list(v) for k, v in loss_plot.items()},
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
                    # Mirror BaseModel.load(): restore non-state_dict attributes
                    # (e.g. sparse synapse_counts, sign flips) so the reloaded
                    # model is truly identical to the one that was saved.
                    if 'extra_state' in checkpoint:
                        model._restore_extra_state(
                            checkpoint['extra_state'], model.device, model.dtype)

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
                    # Keep best_loss unchanged: the reloaded checkpoint already
                    # achieved this loss, so future saves should only fire when
                    # we actually beat it. Resetting to inf would let the very
                    # next epoch (always < inf) clobber the best checkpoint
                    # with a worse one.
                    print(' -------------------------------- ')
                    print(f'Reloaded best checkpoint (loss={best_loss.item():.4f}). '
                          f'Learning rate reduced to {alpha:.2e}.')
                    print(' -------------------------------- ')

        if (epoch % save_every == 0) or (step >= num_steps):
            pu.plot_loss(loss_plot, save_folder=model_save_folder)

        epoch_loss_str = f'{epoch_avg.item():.4f}' if epoch_losses else 'NaN'
        comp_str = ''
        if epoch_losses:
            comp_parts = []
            for name in all_component_names:
                vals = epoch_component_losses[name]
                if vals:
                    avg = torch.stack(vals).mean().item()
                    comp_parts.append(f'{name}={avg:.4f}')
            comp_str = ' (' + ', '.join(comp_parts) + ')'
        print(f'epoch {epoch} | steps {step}/{num_steps} | loss = {epoch_loss_str}{comp_str} | dt = {time.time()-epoch_start:.3g}s | lr = {alpha:.2e} | path = {model_save_folder}')

    return model, loss_plot, opt.state_dict(), {
        "best_loss": float(best_loss.item()),
        "alpha": alpha,
        "time_since_best": time_since_best,
        "last_reload_state": last_reload_state,
        "last_reload_opt_state": last_reload_opt_state,
    }
