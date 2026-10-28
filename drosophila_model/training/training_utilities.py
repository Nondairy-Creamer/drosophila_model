import torch
from torch.utils.data import Dataset, DataLoader
from drosophila_model.training import plotting_utilities as pu
import numpy as np
import inspect
import time
import tempfile
import os
import subprocess
from pathlib import Path

from drosophila_model.config import get_data_folder
from drosophila_model.training import loss_functions as lf

# Call signature shared by every loss component in loss_functions.py.
_LOSS_COMPONENT_ARGS = ['model', 'outputs', 'target', 'train_mask', 'loss_start_step']


def get_git_commit_hash():
    """Get the repo's current git commit hash, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def load_processed_stimulus(config, device=None, dtype=None, split='train'):
    """Load pre-baked stimulus dataset(s).

    Reads ``config['stimulus']['stimulus_type']`` (a string or list of
    strings), loads each file from
    ``<data_folder>/stimulus/training/<type>_<split>.pt`` (produced by
    ``create_training_data.py``), validates that all share the same eye
    count, camera offset and time step, and concatenates them along the
    example axis. Stimuli are not generated on the fly during training.

    The train/test split is baked into separate ``_train.pt`` / ``_test.pt``
    files by ``create_training_data.py``: a seeded shuffle of whole source
    groups, the first ``train_fraction`` of which go to training
    (``[data].shared.train_fraction``, which a type's own ``[data.<type>]``
    table may override). Training loads ``split='train'``; evaluation scripts
    load ``split='test'``. Types baked train-only (``train_fraction = 1.0``,
    e.g. ``full_field_step``) have no ``_test.pt`` and are skipped, with a
    note, when loading the test split; it is an error only if none of the
    requested types has one.

    Each file must have been baked at ``config['model']['dt']``, and a list of
    types must share one example shape (time steps x columns); a clear error
    asking for a re-bake is raised otherwise.

    Returns a dict with keys ``stimulus``, ``optic_flow``, ``train_mask``,
    ``rotation``, ``translation``, ``num_eyes``, ``eye_offset``.
    ``rotation``/``translation`` are ``None`` for one-eye datasets.
    """
    if split not in ('train', 'test'):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if device is None:
        device = torch.device(config['model']['device'])
    if dtype is None:
        dtype = getattr(torch, config['model']['dtype'])

    training_dir = get_data_folder(config) / 'stimulus' / 'training'

    stimulus_type = config['stimulus']['stimulus_type']
    stimulus_types = [stimulus_type] if isinstance(stimulus_type, str) else list(stimulus_type)

    stim_list, flow_list, mask_list, rot_list, trans_list = [], [], [], [], []
    num_eyes_set, eye_offsets = set(), set()
    loaded_types = []
    for stim_type in stimulus_types:
        training_path = training_dir / f'{stim_type}_{split}.pt'
        if not training_path.exists():
            train_path = training_dir / f'{stim_type}_train.pt'
            if split == 'test' and train_path.exists():
                if _is_train_only_bake(train_path):
                    print(f"Skipping '{stim_type}': it was baked train-only "
                          f"(train_fraction = 1.0), so it has no test split.")
                    continue
                raise FileNotFoundError(
                    f"'{stim_type}' has {train_path.name} but no {training_path.name}: "
                    "either every source group went to training (too few groups for "
                    "its train_fraction) or the test file was not copied. Re-bake it "
                    "with: python -m drosophila_model.data_handling.create_training_data "
                    f"{stim_type}"
                )
            raise FileNotFoundError(
                f"Baked stimulus not found: {training_path}. Generate it first with: "
                f"python -m drosophila_model.data_handling.create_training_data {stim_type}"
            )
        data = torch.load(training_path, map_location='cpu', weights_only=False)
        baked_dt = data.get('model_dt')
        if baked_dt is not None and not np.isclose(baked_dt, config['model']['dt']):
            raise ValueError(
                f"{training_path} was baked at dt={baked_dt} but [model].dt is "
                f"{config['model']['dt']}. Re-bake it with the current config."
            )
        num_eyes_set.add(data['num_eyes'])
        eye_offsets.add(tuple(data['eye_offset']) if data['eye_offset'] is not None else None)
        stim_list.append(data['stimulus'].to(device=device, dtype=dtype))
        flow_list.append(data['optic_flow'].to(device=device, dtype=dtype))
        mask_list.append(data['train_mask'].to(device=device))
        if data['rotation'] is not None:
            rot_list.append(data['rotation'].to(device=device, dtype=dtype))
            trans_list.append(data['translation'].to(device=device, dtype=dtype))
        loaded_types.append(stim_type)

    if not loaded_types:
        raise FileNotFoundError(
            f"None of the stimulus types {stimulus_types} has a baked '{split}' "
            f"split in {training_dir} (they were all baked train-only)."
        )
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
    shapes = {stim_type: tuple(s.shape[1:]) for stim_type, s in zip(loaded_types, stim_list)}
    if len(set(shapes.values())) > 1:
        raise ValueError(
            f"Training stimulus types have different (time, column) shapes: {shapes}. "
            "Bake them with the same [data].shared stim_time / pre_stim_time."
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


def _is_train_only_bake(train_path):
    """Whether ``train_path`` exists and was baked with ``train_fraction = 1.0``
    (``create_training_data`` then writes no ``_test.pt``)."""
    if not train_path.exists():
        return False
    # mmap: only the small header entries are read, not the stimulus tensors.
    payload = torch.load(train_path, map_location='cpu', weights_only=False, mmap=True)
    return payload.get('train_fraction') == 1.0


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


def loss_start_step(train_mask):
    """First time step at which every example in the batch is in the loss.

    Datasets with a pre-stimulus period mask its first steps out of the loss;
    returns 0 when no leading steps are masked.
    """
    time_mask = train_mask.any(dim=-1)
    if time_mask[:, 0].all():
        return 0
    return time_mask.all(dim=0).to(torch.int).argmax().item()


def _reusable_mappings(mappings):
    """``(input_mapping, output_mapping)`` from a ``predict`` result's mappings.

    Output mappings are per-region, while the single combined input mapping
    comes from whichever region actually builds one.
    """
    output_mapping = {name: m['output_mapping'] for name, m in mappings.items()}
    input_mapping = next(
        (m['input_mapping'] for m in mappings.values()
         if m.get('input_mapping') is not None),
        None,
    )
    return input_mapping, output_mapping


def model_outputs(model, inputs, train_mask, num_eyes, eye_offset=None,
                  input_neurons=None, output_neurons=None):
    """Run ``model`` on a batch exactly as training does.

    Returns ``(outputs, mappings)``. ``outputs`` is the flat dict of readouts
    the loss functions consume. For two-eye inputs (``[right | left]`` on the
    column axis) the optic-lobe two-eye ego-motion path is used: readouts of
    both eyes are concatenated on the column axis and ``pred_rotation`` /
    ``pred_translation`` are added, solved with the rig's ``eye_offset``.
    ``mappings`` is ``(input_mapping, output_mapping)`` from
    :meth:`ConnectomeModel.predict`; pass it back as ``input_neurons`` /
    ``output_neurons`` to skip rebuilding the stim<->cell mappings on the next
    call. ``None`` neuron lists use each region's defaults.
    """
    if num_eyes == 2:
        region = model.regions['optic_lobe']
        N = inputs.shape[2] // 2
        oob = ~train_mask.any(dim=1).any(dim=0)
        cached = None
        if input_neurons is not None or output_neurons is not None:
            # Same layout as predict()'s result['mappings']: the region's own
            # output spec (names or readout dict), not the per-region dict.
            if isinstance(output_neurons, dict):
                output_neurons = output_neurons[region.name]
            cached = {region.name: {'input_mapping': input_neurons,
                                    'output_mapping': output_neurons}}
        pred_rot, pred_trans, right_res, left_res = region.estimate_ego_motion_from_image(
            inputs[:, :, :N], inputs[:, :, N:], oob_columns=oob,
            cached_mappings=cached, camera_offset=eye_offset)

        # Concatenate right+left readouts on the column axis for the
        # two-eye loss terms.
        outputs = {}
        right_optic = right_res['outputs'].get(region.name, {})
        left_optic = left_res['outputs'].get(region.name, {})
        for key in right_optic.keys():
            if key in ('voltage', 'firing_rate'):
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
        return outputs, _reusable_mappings(right_res['mappings'])

    result = model.predict(inputs, input_neurons=input_neurons,
                           output_neurons=output_neurons)
    # Flatten every region's readouts into one dict for the loss functions;
    # readout names are unique across regions.
    outputs = {}
    for region_outputs in result['outputs'].values():
        outputs.update(region_outputs)
    return outputs, _reusable_mappings(result['mappings'])


def _is_loss_component(name):
    """Whether ``loss_functions.<name>`` has the loss-component signature."""
    fn = getattr(lf, name, None)
    return (inspect.isfunction(fn) and not name.startswith('_')
            and list(inspect.signature(fn).parameters) == _LOSS_COMPONENT_ARGS)


def check_loss_components(loss_components, num_eyes=None):
    """Raise ``ValueError`` for loss names training cannot use.

    Every name must be a loss component in ``training/loss_functions.py`` (a
    function taking ``(model, outputs, target, train_mask, loss_start_step)``);
    ``odometry_loss`` additionally needs two-eye data (checked when
    ``num_eyes`` is given).
    """
    unknown = [name for name in loss_components if not _is_loss_component(name)]
    if unknown:
        raise ValueError(
            f"Unknown loss component(s) {unknown}; [training].loss_components "
            f"must name loss components in training/loss_functions.py."
        )
    if num_eyes is not None and 'odometry_loss' in loss_components and num_eyes != 2:
        raise ValueError("odometry_loss requires a two-eye dataset (num_eyes=2).")


def compute_loss_components(model, outputs, target, train_mask, loss_components):
    """Per-component training losses for one batch.

    Evaluates each ``loss_functions.<name>`` in ``loss_components`` and adds
    any training-only regularizer readouts present in ``outputs``. The total
    training loss is ``sum(result.values())``.
    """
    start = loss_start_step(train_mask)
    component_vals = {
        name: getattr(lf, name)(model, outputs, target, train_mask, start)
        for name in loss_components
    }
    for name, mod in model._readout_modules.items():
        if getattr(mod, 'training_only', False) and name in outputs:
            component_vals[name] = outputs[name]
    return component_vals


def fit(model, inputs, target, output_neurons, input_neurons,
                loss_components,
                rng=None, model_save_folder=None,
                num_steps=100, save_every=100, alpha=1e-4,
                clip_grad_norm=None, adam_beta2=0.99,
                reload_best_on_loss_jump=True,
                patience=10, alpha_factor=0.5,
                opt_state_dict=None, loss_history=None,
                input_noise_std=0.0, contrast_std=0.0, luminance_std=0.0,
                batch_size=None, train_mask=None, num_eyes=1, eye_offset=None):
    """Train ``model`` on ``inputs`` -> ``target`` with Adam.

    ``num_eyes`` is the eye count of the dataset (``load_processed_stimulus``
    returns it). Two-eye inputs are laid out ``[right | left]`` on the column
    axis and are run through the optic-lobe region's two-eye ego-motion path,
    with ``eye_offset`` as the rig's right-eye camera offset; the readouts of
    both eyes are concatenated for the flow loss. ``odometry_loss`` requires
    two-eye data.

    ``model.pt`` / ``model_opt.pt`` in ``model_save_folder`` are overwritten
    whenever the epoch loss improves, and the loss curve is drawn to
    ``<model_save_folder>/figures/training/loss.png``. ``model_opt.pt`` holds
    the optimizer state and per-component loss history; pass them back as
    ``opt_state_dict`` / ``loss_history`` to resume a run (a resumed run only
    overwrites ``model.pt`` once it beats the best epoch of ``loss_history``).
    ``clip_grad_norm=None`` disables gradient clipping.

    ``adam_beta2`` is Adam's second-moment decay. Training starts on a flat
    stretch (flow_loss ~1) where almost no gradient reaches the network, and
    rare large gradients (the first step, or a spiking batch) land in the
    second-moment estimate. With PyTorch's default 0.999 (~1000-step memory)
    one spike shrinks every step for hundreds of steps and can stall the
    start of learning; 0.99 (~100 steps) lets it recover. A resumed
    ``opt_state_dict`` keeps the betas it was saved with.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    if num_eyes not in (1, 2):
        raise ValueError(f"num_eyes must be 1 or 2, got {num_eyes}")
    check_loss_components(loss_components, num_eyes)

    if model_save_folder is not None:
        model_save_folder = Path(model_save_folder)

    if train_mask is None:
        train_mask = torch.ones(inputs.shape[:3], dtype=torch.bool, device=inputs.device)

    opt = torch.optim.Adam(model.parameters(), lr=alpha, betas=(0.9, adam_beta2))

    # Region-agnostic mapping round-trip. The first iteration passes the
    # caller-provided cell-type-name lists (None = region defaults); predict()
    # builds the per-region stim<->cell mappings and we reuse those on every
    # later iteration (one- and two-eye) so the index work happens only once.
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
    # Resuming a run: model.pt already holds the best epoch of loss_history,
    # so only overwrite it once that loss is beaten.
    history_lengths = {len(v) for v in loss_plot.values()}
    if loss_history and len(history_lengths) == 1 and history_lengths != {0}:
        history_total = np.sum(list(loss_plot.values()), axis=0)
        finite_total = history_total[np.isfinite(history_total)]
        if finite_total.size:
            best_loss = torch.tensor(float(finite_total.min()),
                                     device=model.device, dtype=model.dtype)
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

            outputs, mappings = model_outputs(
                model, inputs_batch, train_mask_batch, num_eyes,
                eye_offset=eye_offset,
                input_neurons=predict_input_neurons,
                output_neurons=predict_output_neurons)
            predict_input_neurons, predict_output_neurons = mappings

            component_vals = compute_loss_components(
                model, outputs, target_batch, train_mask_batch, loss_components)
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
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile('wb', delete=False, dir=str(opt_path.parent), suffix='.tmp') as tmp:
                            tmp_path = tmp.name
                            torch.save(opt_payload, tmp_path)
                            tmp.flush()
                            os.fsync(tmp.fileno())
                        os.replace(tmp_path, str(opt_path))
                    except Exception:
                        try:
                            if tmp_path is not None and os.path.exists(tmp_path):
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
                    # Mirror SerializableModel.load(): restore non-state_dict attributes
                    # (the sparse synapse_counts) so the reloaded
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

        if model_save_folder is not None and ((epoch % save_every == 0) or (step >= num_steps)):
            pu.plot_loss(loss_plot, save_folder=model_save_folder / 'figures' / 'training')

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
