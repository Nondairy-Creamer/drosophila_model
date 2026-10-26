"""
Run a trained model on a freshly generated stimulus and visualize flow.

Defaults to two-eye stimuli; pass ``--no-two-eyes`` for a single eye. The
two-eye path also saves a static PNG of the stimulus' rotational and
translational velocity over time alongside the movie.

One-eye layout (3 rows):
  Row 0: Stimulus intensity (scatter)
  Row 1: True optic flow (quiver)
  Row 2: Predicted optic flow (quiver)

Two-eye layout (3x2):
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left true flow       | Right true flow
  Row 2: Left predicted flow  | Right predicted flow

When --ego-motion is set (two-eye only), the model's ego-motion estimate is
shown with rotation/translation rate trailing-window plots:
  Row 0: Left stimulus        | Right stimulus
  Row 1: Left predicted flow  | Right predicted flow
  Row 2: Rotation rate        | Translation rate

For a two-eye stimulus that carries ground-truth odometry (e.g. --stim-type
flyview), the model's estimated rotation/translation rates are overlaid against
the ground truth. Ego-motion is solved with the dataset's registered camera
offset, as in training (two-eye ``get_stimulus`` data goes through
``training_utilities.model_outputs``).

Columns outside the field of view are drawn yellow.

Usage (from the repo root):
    python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/my_model/ --no-two-eyes --stim-type random_sine
    python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/my_model/ --stim-type random_pano
    python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/my_model/ --ego-motion --stim-type random_pano
    python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/my_model/ --ego-motion --stim-type flyview
    python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/my_model/ --ego-motion --translation

``--duration <seconds>`` overrides the generated stimulus length
(``[shared].stim_time`` in the data config); ``--max-frames`` is a separate hard
cap on how many frames get rendered.
"""
import argparse
from pathlib import Path


import numpy as np
import torch

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.model_definition import neural_models as nm
from drosophila_model.data_handling import stimulus_generators as su
from drosophila_model.training import training_utilities as tu
from drosophila_model.analysis.model_analysis.analysis_utilities import (
    cos_sim, save_integrated_figure, save_rate_figure,
)

from drosophila_model.analysis.movie_scripts.movie_utilities import (
    masked_rgba,
    write_movie,
    ylim_from_data,
    get_fov,
    subtract_rotation_flow,
    synthesize_two_eye_flow_from_odometry,
    get_flow_key,
    prepare_eye_quiver,
    setup_eye_axis,
    init_eye_quiver,
    update_eye_quiver,
    add_eye_colorwheel,
)


def _display_mask(train_mask_np, pre_stim_steps):
    """Field-of-view mask for display from an ``(examples, T, columns)`` train mask.

    The leading pre-stimulus frames are excluded from the loss
    (``train_mask`` False) but are on screen; show them with the field of view
    of the first stimulus frame instead of painting every column as masked.
    """
    mask = train_mask_np.astype(bool)
    mask[:, :pre_stim_steps] = mask[:, pre_stim_steps:pre_stim_steps + 1]
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model on a freshly generated stimulus and visualize flow."
    )
    parser.add_argument("model_folder", type=str, help="Path to model folder or model.pt")
    parser.add_argument("--config-path", "--config", dest="config_path", type=str,
                        default="config.toml",
                        help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--data-config-path", type=str, default="config_data.toml",
                        help="Path to the stimulus-generation config overlay "
                             "(default: ./config_data.toml)")
    parser.add_argument("--two-eyes", action=argparse.BooleanOptionalAction, default=True,
                        help="Use two-eye stimulus and predict_two_eyes (default: two eyes; "
                             "pass --no-two-eyes for a single eye)")
    parser.add_argument("--stim-type", type=str, default=None,
                        help="Stimulus type (default: random_pano for 2-eye, random_sine for 1-eye)")
    parser.add_argument("--sine-direction", type=str, nargs="+", default=None,
                        choices=["right", "left", "up", "down"],
                        help="Direction(s) for directional_sine (default: all four)")
    parser.add_argument("--stim-index", type=int, nargs="+", default=[0],
                        help="Stimulus indices to visualize (space-separated)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output movie frame rate (default: the model's base rate)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Generated stimulus duration in seconds (overrides "
                             "[shared].stim_time from the data config).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Hard cap on the number of frames to render")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/figures/run_model_on_stimulus)")
    parser.add_argument("--ego-motion", action="store_true",
                        help="Estimate ego-motion from the two-eye flow and show rate plots")
    parser.add_argument("--translation", action="store_true",
                        help="Subtract rotation-induced flow (requires --ego-motion)")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save rate and integrated pose PNG figures (requires --ego-motion)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Stimulus-generation seed (overrides [shared].seed "
                             "from the data config)")
    args = parser.parse_args()

    if args.stim_type is None:
        args.stim_type = "random_pano" if args.two_eyes else "random_sine"

    if args.ego_motion and not args.two_eyes:
        parser.error("--ego-motion requires --two-eyes")
    if args.stim_type == "directional_sine" and args.two_eyes:
        parser.error("--stim-type directional_sine is rendered per direction on one eye; "
                     "pass --no-two-eyes")
    if (args.translation or args.save_figures) and not args.ego_motion:
        parser.error("--translation and --save-figures require --ego-motion")

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path

    config = load_config(args.config_path, args.data_config_path)
    data_config = config['data']
    if args.duration is not None:
        if args.duration <= 0:
            parser.error("--duration must be positive")
        print(f"Overriding stimulus duration {data_config['shared']['stim_time']}s "
              f"-> {args.duration}s")
        data_config['shared']['stim_time'] = args.duration
    if args.seed is not None:
        data_config['shared']['seed'] = args.seed
    model = nm.BaseRnn.load(model_path)

    device = model.device
    dtype = model.dtype
    model_dt = model.dt
    column_phi_theta = model.regions['optic_lobe'].column_phi_theta
    N = column_phi_theta.shape[0]

    # Force stimulus generation to use the model's dt and device so the resampled
    # stimulus rate matches the movie playback rate, and device matches the model.
    cfg_dt = float(config['model']['dt'])
    if abs(cfg_dt - model_dt) > 1e-9:
        print(f"Overriding config dt {cfg_dt} -> model dt {model_dt} for stimulus generation")
        config['model']['dt'] = model_dt
    cfg_device = str(config['model'].get('device', ''))
    if cfg_device != str(device):
        print(f"Overriding config device {cfg_device!r} -> model device {str(device)!r}")
        config['model']['device'] = str(device)

    frame_rate = args.fps or max(1, round(1 / model_dt))

    column_xyz = model.regions['optic_lobe'].get_column_xyz()
    cell_x, cell_y, cell_z = column_xyz[:, 0], column_xyz[:, 1], column_xyz[:, 2]

    flow_key = get_flow_key(model)
    if flow_key is None:
        raise SystemExit("Model has no flow readout (predicts_flow=True); "
                         "cannot render a flow movie.")

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "figures" / "run_model_on_stimulus")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.two_eyes and args.ego_motion and args.stim_type == "random_pano":
        _run_ego_motion_pano(args, model, data_config, get_data_folder(config), output_dir,
                             device, dtype, model_dt, column_phi_theta, N,
                             cell_x, cell_y, cell_z, frame_rate, flow_key)
    elif args.two_eyes and args.ego_motion:
        _run_ego_motion_two_eyes(args, model, config, data_config, output_dir,
                                 device, dtype, model_dt, column_phi_theta, N,
                                 cell_x, cell_y, cell_z, frame_rate, flow_key)
    elif args.two_eyes:
        _run_two_eyes(args, model, config, data_config, output_dir,
                      device, dtype, model_dt, column_phi_theta, N,
                      cell_x, cell_y, cell_z, frame_rate, flow_key)
    else:
        _run_one_eye(args, model, config, data_config, output_dir,
                     device, dtype, model_dt, column_phi_theta,
                     cell_x, cell_y, cell_z, frame_rate, flow_key)


def _pre_stim_steps(data_config, model_dt):
    """Number of pre-stimulus frames ``get_stimulus`` prepends."""
    return int(round(data_config['shared']['pre_stim_time'] / model_dt))


def _two_eye_outputs(model, data, stim_index, num_frames, flow_key, N, device, dtype):
    """Run one two-eye example of ``data`` through the model as training does.

    Uses :func:`training_utilities.model_outputs`: the two-eye ego-motion path,
    with columns never in the example's field of view excluded and the
    dataset's camera offset. Returns numpy ``(flow_right, flow_left,
    rotation, translation)`` for the first ``num_frames`` frames; the left-eye
    flow is in the shared frame.
    """
    inputs = data.stimulus[stim_index:stim_index + 1, :num_frames].to(device=device, dtype=dtype)
    # Full-length mask: out-of-FOV columns are those never valid in the example.
    train_mask = data.train_mask[stim_index:stim_index + 1].to(device=device)
    with torch.no_grad():
        outputs, _ = tu.model_outputs(model, inputs, train_mask, num_eyes=2,
                                      eye_offset=data.eye_offset)
    flow = outputs[flow_key][0].cpu().numpy()
    return (flow[:, :N], flow[:, N:],
            outputs['pred_rotation'][0].cpu().numpy(),
            outputs['pred_translation'][0].cpu().numpy())


# ---------------------------------------------------------------------------
# One-eye path
# ---------------------------------------------------------------------------

def _run_one_eye(args, model, config, data_config, output_dir,
                 device, dtype, model_dt, column_phi_theta,
                 cell_x, cell_y, cell_z, frame_rate, flow_key):
    pre_stim_steps = _pre_stim_steps(data_config, model_dt)
    if args.stim_type == 'directional_sine':
        directions = args.sine_direction or ['right', 'left', 'up', 'down']
        stim_parts, flow_parts, mask_parts = [], [], []
        for d in directions:
            print(f"Generating stimulus (type={args.stim_type}, direction={d})...")
            data = su.get_stimulus(
                args.stim_type, config, column_phi_theta, num_eyes=1,
                data_config=data_config, sine_direction=d)
            s, f, m = data.stimulus, data.optic_flow, data.train_mask
            stim_parts.append(s)
            flow_parts.append(f)
            mask_parts.append(m)

        n_per_dir = stim_parts[0].shape[0]

        for stim_index in args.stim_index:
            if stim_index >= n_per_dir:
                print(f"Skipping stim {stim_index}: only {n_per_dir} examples per direction")
                continue

            batch_stim = torch.stack(
                [sp[stim_index] for sp in stim_parts]).to(device=device, dtype=dtype)
            batch_true = np.stack(
                [fp[stim_index].cpu().numpy() for fp in flow_parts])
            batch_mask = _display_mask(np.stack(
                [mp[stim_index].cpu().numpy() for mp in mask_parts]), pre_stim_steps)

            num_frames = batch_stim.shape[1]
            if args.max_frames is not None:
                num_frames = min(num_frames, args.max_frames)
                batch_stim = batch_stim[:, :num_frames]
                batch_true = batch_true[:, :num_frames]
                batch_mask = batch_mask[:, :num_frames]

            print(f"\nRunning model on stimulus {stim_index}, "
                  f"{len(directions)} directions ({num_frames} frames)...")
            with torch.no_grad():
                result = model.predict(batch_stim)
                batch_pred = result["outputs"]["optic_lobe"][flow_key].cpu().numpy()[:, :num_frames]

            out_path = output_dir / f"directional_sine_stim_{stim_index}.mp4"
            print(f"  Creating movie -> {out_path}")
            _make_one_eye_directional_movie(
                stimuli=batch_stim.cpu().numpy(),
                true_flows=batch_true,
                pred_flows=batch_pred,
                directions=directions,
                cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
                output_path=out_path, frame_rate=frame_rate,
                masks=batch_mask,
            )
        return

    print(f"Generating stimulus (type={args.stim_type})...")
    data = su.get_stimulus(
        args.stim_type, config, column_phi_theta, num_eyes=1, data_config=data_config,
    )
    stimulus, optic_flow, train_mask = data.stimulus, data.optic_flow, data.train_mask
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    display_mask = _display_mask(train_mask.cpu().numpy(), pre_stim_steps)
    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_input = stimulus[stim_index:stim_index + 1].to(device=device, dtype=dtype)
        true_flow = optic_flow_np[stim_index]
        mask = display_mask[stim_index]

        num_frames = stim_input.shape[1]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_input = stim_input[:, :num_frames]
            true_flow = true_flow[:num_frames]
            mask = mask[:num_frames]

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames)...")
        with torch.no_grad():
            result = model.predict(stim_input)
            pred_flow = result["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()

        pred_flow = pred_flow[:num_frames]
        stim_np = stimulus_np[stim_index, :num_frames]

        # "_one_eye" keeps it apart from the two-eye movie of the same stimulus.
        out_path = output_dir / f"{args.stim_type}_one_eye_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_one_eye_flow_movie(
            stimulus=stim_np, true_flow=true_flow, pred_flow=pred_flow,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate,
            mask=mask, stim_type=args.stim_type,
        )


# ---------------------------------------------------------------------------
# Two-eye path
# ---------------------------------------------------------------------------

def _run_two_eyes(args, model, config, data_config, output_dir,
                  device, dtype, model_dt, column_phi_theta, N,
                  cell_x, cell_y, cell_z, frame_rate, flow_key):
    print(f"Generating two-eye stimulus (type={args.stim_type})...")
    data = su.get_stimulus(args.stim_type, config, column_phi_theta, num_eyes=2,
                           data_config=data_config)
    stimulus, optic_flow = data.stimulus, data.optic_flow
    observer_rotation, observer_translation, train_mask = (
        data.rotation, data.translation, data.train_mask)
    stimulus_np = stimulus.cpu().numpy()
    optic_flow_np = optic_flow.cpu().numpy()
    observer_rotation_np = observer_rotation.cpu().numpy()
    observer_translation_np = observer_translation.cpu().numpy()
    pre_stim_steps = _pre_stim_steps(data_config, model_dt)
    display_mask = _display_mask(train_mask.cpu().numpy(), pre_stim_steps)
    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    phi = column_phi_theta[:, 0]
    theta = column_phi_theta[:, 1]

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_right = stimulus_np[stim_index, :, :N]
        stim_left = stimulus_np[stim_index, :, N:]
        flow_right = optic_flow_np[stim_index, :, :N, :]
        flow_left = optic_flow_np[stim_index, :, N:, :]
        mask_right = display_mask[stim_index, :, :N]
        mask_left = display_mask[stim_index, :, N:]

        if np.all(np.isnan(flow_right[pre_stim_steps:])):
            print(f"  Stim {stim_index}: ground-truth flow unavailable - "
                  "synthesizing from odometry (Z=1)")
            flow_right, flow_left = synthesize_two_eye_flow_from_odometry(
                observer_rotation_np[stim_index], observer_translation_np[stim_index],
                phi, theta, eye_offset=data.eye_offset)
            flow_label = "Flow (from odometry, Z=1)"
        else:
            flow_label = "True Flow"

        num_frames = stim_right.shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stim_right = stim_right[:num_frames]
            stim_left = stim_left[:num_frames]
            flow_right = flow_right[:num_frames]
            flow_left = flow_left[:num_frames]
            mask_right = mask_right[:num_frames]
            mask_left = mask_left[:num_frames]

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames, N={N})...")
        pred_flow_right, pred_flow_left, pred_rot_np, pred_trans_np = _two_eye_outputs(
            model, data, stim_index, num_frames, flow_key, N, device, dtype)

        out_path = output_dir / f"{args.stim_type}_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_two_eye_flow_movie(
            stimulus_left=stim_left, stimulus_right=stim_right,
            true_flow_left=flow_left, true_flow_right=flow_right,
            pred_flow_left=pred_flow_left, pred_flow_right=pred_flow_right,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            output_path=out_path, frame_rate=frame_rate,
            mask_left=mask_left, mask_right=mask_right,
            stim_type=args.stim_type, flow_label=flow_label,
        )

        # Overlay the model's estimated ego-motion (dashed) on the stimulus'
        # ground-truth rotational/translational velocity.
        true_rot = observer_rotation_np[stim_index, :num_frames]
        true_trans = observer_translation_np[stim_index, :num_frames]
        # Translation is recovered only up to scale; match GT magnitude for the
        # overlay when the stimulus actually translates (else show raw prediction).
        if np.std(true_trans) > 1e-9:
            pred_trans_np = pred_trans_np * (np.std(true_trans) / (np.std(pred_trans_np) + 1e-12))
        save_rate_figure(
            true_rot, pred_rot_np, true_trans, pred_trans_np,
            model_dt, f"Stimulus {stim_index}",
            output_dir / f"{args.stim_type}_velocity_stim_{stim_index}.png",
        )


# ---------------------------------------------------------------------------
# Ego-motion pano path (random_pano with per-frame out-of-panorama masks)
# ---------------------------------------------------------------------------

def _run_ego_motion_pano(args, model, data_config, data_folder, output_dir,
                         device, dtype, model_dt, column_phi_theta, N,
                         cell_x, cell_y, cell_z, frame_rate, flow_key):
    """Ego-motion movie for ``random_pano`` (pure rotation, zero translation).

    Generates the panoramas directly (rather than through ``get_stimulus``) to
    keep the per-frame out-of-panorama mask, which drives the field-of-view
    display, the excluded solver columns and the true flow magnitude.
    ``data_folder`` holds ``stimulus/baked/all_panoramic.pt``.
    """
    both_phi = np.concatenate([column_phi_theta[:, 0], -column_phi_theta[:, 0]])
    both_theta = np.concatenate([column_phi_theta[:, 1], column_phi_theta[:, 1]])

    pano_params = data_config['random_pano']
    stim_time = data_config['shared']['stim_time']
    seed = data_config['shared']['seed']
    velocity_amp = pano_params['velocity_amp']
    velocity_time_scale = pano_params['velocity_time_scale']
    pre_stim_time = data_config['shared']['pre_stim_time']

    # Examples are pano-major, 2 (symmetrised) repeats per panorama, and the
    # random draws are made pano by pano: generating only the panoramas the
    # requested indices need gives the same examples as generating them all.
    num_panos = min(max(args.stim_index) // 2 + 1,
                    len(su._load_all_panos(data_folder=data_folder)))
    print(f"Generating random pano stimulus for {num_panos} panorama(s) "
          f"(stim_time={stim_time}s, velocity_amp={velocity_amp}, "
          f"velocity_time_scale={velocity_time_scale}, seed={seed})...")
    stimulus_both, optic_flow_both, rotation_vectors, oob_mask = su.get_random_velocity_pano(
        both_phi, both_theta,
        velocity_amp=velocity_amp,
        velocity_time_scale=velocity_time_scale,
        num_repeats=1, dt=model_dt, stim_time=stim_time, seed=seed,
        data_folder=data_folder, pano_indices=list(range(num_panos)),
    )
    n_examples = stimulus_both.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_both.shape}")
    # random_pano is a single-viewpoint render: its registered camera offset.
    camera_offset = su.camera_offset_for_stimulus('random_pano')
    mag_key = next(
        (k for k, m in model._readout_modules.items()
         if getattr(m, 'predicts_motion_mag', False)),
        None,
    )
    pre_stim_frames = int(round(pre_stim_time / model_dt))

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim = stimulus_both[stim_index]
        oob = oob_mask[stim_index]

        true_rot_rate = rotation_vectors[stim_index] * (180.0 / np.pi)
        true_trans_rate = np.zeros_like(true_rot_rate)

        stimulus_right_np = stim[:, :N]
        stimulus_left_np = stim[:, N:]
        oob_right = oob[:, :N]
        oob_left = oob[:, N:]

        flow_both = optic_flow_both[stim_index]
        true_mag_right = np.linalg.norm(flow_both[:, :N, :], axis=-1)
        true_mag_left = np.linalg.norm(flow_both[:, N:, :], axis=-1)

        if pre_stim_frames > 0:
            pre_right = np.full((pre_stim_frames, N), 0.5)
            pre_left = np.full((pre_stim_frames, N), 0.5)
            stimulus_right_np = np.concatenate([pre_right, stimulus_right_np], axis=0)
            stimulus_left_np = np.concatenate([pre_left, stimulus_left_np], axis=0)
            pre_rot = np.zeros((pre_stim_frames, 3))
            true_rot_rate = np.concatenate([pre_rot, true_rot_rate], axis=0)
            true_trans_rate = np.concatenate([pre_rot, true_trans_rate], axis=0)
            pre_oob = np.zeros((pre_stim_frames, N), dtype=bool)
            oob_right = np.concatenate([pre_oob, oob_right], axis=0)
            oob_left = np.concatenate([pre_oob, oob_left], axis=0)
            pre_mag = np.zeros((pre_stim_frames, N))
            true_mag_right = np.concatenate([pre_mag, true_mag_right], axis=0)
            true_mag_left = np.concatenate([pre_mag, true_mag_left], axis=0)

        num_frames = stimulus_right_np.shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
            stimulus_right_np = stimulus_right_np[:num_frames]
            stimulus_left_np = stimulus_left_np[:num_frames]
            true_rot_rate = true_rot_rate[:num_frames]
            true_trans_rate = true_trans_rate[:num_frames]
            oob_right = oob_right[:num_frames]
            oob_left = oob_left[:num_frames]
            true_mag_right = true_mag_right[:num_frames]
            true_mag_left = true_mag_left[:num_frames]

        right_input = torch.tensor(stimulus_right_np, device=device, dtype=dtype).unsqueeze(0)
        left_input = torch.tensor(stimulus_left_np, device=device, dtype=dtype).unsqueeze(0)

        oob_always_right = np.all(oob_right, axis=0)
        oob_always_left = np.all(oob_left, axis=0)
        oob_columns = torch.tensor(
            np.concatenate([oob_always_right, oob_always_left]), device=device)

        print(f"\nRunning model on stimulus {stim_index} ({num_frames} frames, N={N})...")
        with torch.no_grad():
            pred_rot, pred_trans, right_res, left_res = model.regions['optic_lobe'].estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob_columns, camera_offset=camera_offset)

        pred_flow_right = right_res["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()
        pred_flow_left = left_res["outputs"]["optic_lobe"][flow_key][0].cpu().numpy()
        pred_rot_np = pred_rot[0].cpu().numpy()
        pred_trans_np = pred_trans[0].cpu().numpy()

        if args.translation:
            pred_flow_right, pred_flow_left = subtract_rotation_flow(
                pred_flow_right, pred_flow_left, pred_rot_np, column_phi_theta)

        if args.save_figures:
            title = f"Stimulus {stim_index}"
            save_rate_figure(true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
                             model_dt, title,
                             output_dir / f"{args.stim_type}_rate_stim_{stim_index}.png")
            save_integrated_figure(true_rot_rate, pred_rot_np, true_trans_rate, pred_trans_np,
                                   model_dt, title,
                                   output_dir / f"{args.stim_type}_integrated_stim_{stim_index}.png")

        rot_cos = cos_sim(true_rot_rate[pre_stim_frames:], pred_rot_np[pre_stim_frames:])
        print(f"  Rotation cosine similarity: {rot_cos:.4f}")

        in_fov_right = ~oob_right
        in_fov_left = ~oob_left

        flow_label = "translation flow" if args.translation else "predicted flow"
        out_path = output_dir / f"{args.stim_type}_ego_motion_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_ego_motion_movie(
            stimulus_left=stimulus_left_np, stimulus_right=stimulus_right_np,
            pred_flow_left=pred_flow_left[:num_frames],
            pred_flow_right=pred_flow_right[:num_frames],
            true_rot=true_rot_rate, pred_rot=pred_rot_np,
            true_trans=true_trans_rate, pred_trans=pred_trans_np,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            in_fov_left=in_fov_left, in_fov_right=in_fov_right,
            output_path=out_path, frame_rate=frame_rate,
            model_dt=model_dt, flow_label=flow_label,
        )

        if mag_key is not None and args.save_figures:
            pred_mag_right = right_res['outputs']['optic_lobe'][mag_key][0].cpu().numpy()
            pred_mag_left = left_res['outputs']['optic_lobe'][mag_key][0].cpu().numpy()
            _save_magnitude_figure(
                true_mag_right, true_mag_left,
                pred_mag_right, pred_mag_left,
                model_dt, output_dir / f"{args.stim_type}_magnitude_stim_{stim_index}.png")


# ---------------------------------------------------------------------------
# Ego-motion path for two-eye stimuli with ground-truth odometry (e.g. flyview)
# ---------------------------------------------------------------------------

def _run_ego_motion_two_eyes(args, model, config, data_config, output_dir,
                             device, dtype, model_dt, column_phi_theta, N,
                             cell_x, cell_y, cell_z, frame_rate, flow_key):
    """Run ego-motion estimation on a generated two-eye stimulus that carries
    ground-truth odometry (e.g. ``flyview``) and compare true vs predicted rates.

    Generates the stimulus with :func:`get_stimulus` (num_eyes=2), estimates
    ego-motion as training does (:func:`_two_eye_outputs`), and renders a
    stimulus + predicted-flow + rotation/translation-rate movie.
    """
    print(f"Generating two-eye stimulus (type={args.stim_type})...")
    data = su.get_stimulus(args.stim_type, config, column_phi_theta, num_eyes=2,
                           data_config=data_config)
    stimulus_np = data.stimulus.cpu().numpy()
    observer_rotation_np = data.rotation.cpu().numpy()
    observer_translation_np = data.translation.cpu().numpy()
    display_mask = _display_mask(data.train_mask.cpu().numpy(),
                                 _pre_stim_steps(data_config, model_dt))
    n_examples = stimulus_np.shape[0]
    print(f"  Generated {n_examples} stimulus examples, shape {stimulus_np.shape}")

    for stim_index in args.stim_index:
        if stim_index >= n_examples:
            print(f"Skipping stim {stim_index}: only {n_examples} examples available")
            continue

        stim_right = stimulus_np[stim_index, :, :N]
        stim_left = stimulus_np[stim_index, :, N:]
        mask_right = display_mask[stim_index, :, :N]
        mask_left = display_mask[stim_index, :, N:]

        num_frames = stim_right.shape[0]
        if args.max_frames is not None:
            num_frames = min(num_frames, args.max_frames)
        stim_right = stim_right[:num_frames]
        stim_left = stim_left[:num_frames]
        mask_right = mask_right[:num_frames]
        mask_left = mask_left[:num_frames]

        print(f"\nEstimating ego-motion for stimulus {stim_index} "
              f"({num_frames} frames, N={N})...")
        pred_flow_right, pred_flow_left, pred_rot_np, pred_trans_np = _two_eye_outputs(
            model, data, stim_index, num_frames, flow_key, N, device, dtype)

        if args.translation:
            pred_flow_right, pred_flow_left = subtract_rotation_flow(
                pred_flow_right, pred_flow_left, pred_rot_np, column_phi_theta)

        true_rot = observer_rotation_np[stim_index][:num_frames]
        true_trans = observer_translation_np[stim_index][:num_frames]
        # Translation is recovered only up to scale; match GT magnitude for the
        # overlay when the stimulus actually translates (else show raw prediction).
        if np.std(true_trans) > 1e-9:
            pred_trans_np = pred_trans_np * (np.std(true_trans) / (np.std(pred_trans_np) + 1e-12))

        print(f"  Rotation cosine similarity:    {cos_sim(true_rot, pred_rot_np):.4f}")
        print(f"  Translation cosine similarity: {cos_sim(true_trans, pred_trans_np):.4f}")

        if args.save_figures:
            title = f"Stimulus {stim_index}"
            save_rate_figure(true_rot, pred_rot_np, true_trans, pred_trans_np,
                             model_dt, title, output_dir / f"{args.stim_type}_rate_stim_{stim_index}.png")
            save_integrated_figure(true_rot, pred_rot_np, true_trans, pred_trans_np,
                                   model_dt, title, output_dir / f"{args.stim_type}_integrated_stim_{stim_index}.png")

        out_path = output_dir / f"{args.stim_type}_ego_motion_stim_{stim_index}.mp4"
        print(f"  Creating movie -> {out_path}")
        _make_ego_motion_movie(
            stimulus_left=stim_left, stimulus_right=stim_right,
            pred_flow_left=pred_flow_left, pred_flow_right=pred_flow_right,
            true_rot=true_rot, pred_rot=pred_rot_np,
            true_trans=true_trans, pred_trans=pred_trans_np,
            cell_x=cell_x, cell_y=cell_y, cell_z=cell_z,
            in_fov_left=mask_left, in_fov_right=mask_right,
            output_path=out_path, frame_rate=frame_rate,
            model_dt=model_dt,
            flow_label="translation flow" if args.translation else "predicted flow",
        )


# ---------------------------------------------------------------------------
# Movie rendering functions
# ---------------------------------------------------------------------------

def _make_one_eye_flow_movie(
    stimulus, true_flow, pred_flow, cell_x, cell_y, cell_z,
    output_path, frame_rate, mask=None, stim_type="",
):
    """Render a 3-row movie: stimulus, true flow, predicted flow."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus.shape[0]
    gray_cmap = plt.cm.gray
    stim_norm = mcolors.Normalize(vmin=np.nanmin(stimulus), vmax=np.nanmax(stimulus))

    # Single-eye renderers default to the right-eye view (matches prior behavior).
    eye = "right"
    true_vx, true_vz, _, _, _ = prepare_eye_quiver(true_flow, cell_x, cell_y, cell_z, eye)
    pred_vx, pred_vz, _, _, _ = prepare_eye_quiver(pred_flow, cell_x, cell_y, cell_z, eye)

    all_mag = np.hypot(
        np.concatenate([true_vx.ravel(), pred_vx.ravel()]),
        np.concatenate([true_vz.ravel(), pred_vz.ravel()]),
    )
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(8, 16), dpi=100)
    ax_stim, ax_true, ax_pred = axes

    for ax in axes:
        setup_eye_axis(ax, cell_x, cell_z, eye)

    mask_0 = mask[0] if mask is not None else None
    sc_stim = ax_stim.scatter(
        cell_x, cell_z, c=masked_rgba(stimulus[0], mask_0, gray_cmap, stim_norm),
        s=90, edgecolors="none")
    ax_stim.set_title(f"Stimulus ({stim_type})")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    q_true = init_eye_quiver(
        ax_true, cell_x, cell_z, true_vx[0], true_vz[0], max_mag, eye, **quiver_kw)
    ax_true.set_title("True Flow")

    q_pred = init_eye_quiver(
        ax_pred, cell_x, cell_z, pred_vx[0], pred_vz[0], max_mag, eye, **quiver_kw)
    ax_pred.set_title("Predicted Flow")
    plt.tight_layout()
    add_eye_colorwheel(fig)  # after tight_layout: the polar wheel is not a subplot

    def update(t):
        mask_t = mask[t] if mask is not None else None
        sc_stim.set_facecolors(masked_rgba(stimulus[t], mask_t, gray_cmap, stim_norm))
        update_eye_quiver(q_true, true_vx[t], true_vz[t], max_mag, eye)
        update_eye_quiver(q_pred, pred_vx[t], pred_vz[t], max_mag, eye)
        ax_stim.set_title(f"Stimulus ({stim_type}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_one_eye_directional_movie(
    stimuli, true_flows, pred_flows, directions,
    cell_x, cell_y, cell_z,
    output_path, frame_rate, masks=None,
):
    """Render a 3×D grid movie: stimulus / true flow / pred flow per direction."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    n_dirs = len(directions)
    num_frames = stimuli.shape[1]
    gray_cmap = plt.cm.gray
    stim_norm = mcolors.Normalize(vmin=np.nanmin(stimuli), vmax=np.nanmax(stimuli))

    eye = "right"
    true_quivers = []
    pred_quivers = []
    for d_idx in range(n_dirs):
        tvx, tvz, _, _, _ = prepare_eye_quiver(true_flows[d_idx], cell_x, cell_y, cell_z, eye)
        pvx, pvz, _, _, _ = prepare_eye_quiver(pred_flows[d_idx], cell_x, cell_y, cell_z, eye)
        true_quivers.append((tvx, tvz))
        pred_quivers.append((pvx, pvz))

    all_vx = np.concatenate([q[0].ravel() for q in true_quivers + pred_quivers])
    all_vz = np.concatenate([q[1].ravel() for q in true_quivers + pred_quivers])
    all_mag = np.hypot(all_vx, all_vz)
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, n_dirs, figsize=(4 * n_dirs, 12), dpi=100,
                             squeeze=False)

    dot_size = max(20, int(90 * 2 / n_dirs))
    for ax in axes.flat:
        setup_eye_axis(ax, cell_x, cell_z, eye)

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    sc_stims = []
    q_trues = []
    q_preds = []

    for d_idx, d in enumerate(directions):
        mask_0 = masks[d_idx, 0] if masks is not None else None
        sc = axes[0, d_idx].scatter(cell_x, cell_z,
            c=masked_rgba(stimuli[d_idx, 0], mask_0, gray_cmap, stim_norm),
            s=dot_size, edgecolors="none")
        sc_stims.append(sc)
        axes[0, d_idx].set_title(f"Stimulus ({d})")

        tvx, tvz = true_quivers[d_idx]
        qt = init_eye_quiver(
            axes[1, d_idx], cell_x, cell_z, tvx[0], tvz[0], max_mag, eye, **quiver_kw)
        q_trues.append(qt)
        axes[1, d_idx].set_title(f"True Flow ({d})")

        pvx, pvz = pred_quivers[d_idx]
        qp = init_eye_quiver(
            axes[2, d_idx], cell_x, cell_z, pvx[0], pvz[0], max_mag, eye, **quiver_kw)
        q_preds.append(qp)
        axes[2, d_idx].set_title(f"Predicted Flow ({d})")

    plt.tight_layout()
    add_eye_colorwheel(fig)  # after tight_layout: the polar wheel is not a subplot

    def update(t):
        for d_idx in range(n_dirs):
            mask_t = masks[d_idx, t] if masks is not None else None
            sc_stims[d_idx].set_facecolors(
                masked_rgba(stimuli[d_idx, t], mask_t, gray_cmap, stim_norm))
            tvx, tvz = true_quivers[d_idx]
            update_eye_quiver(q_trues[d_idx], tvx[t], tvz[t], max_mag, eye)
            pvx, pvz = pred_quivers[d_idx]
            update_eye_quiver(q_preds[d_idx], pvx[t], pvz[t], max_mag, eye)
        axes[0, 0].set_title(
            f"Stimulus ({directions[0]}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_two_eye_flow_movie(
    stimulus_left, stimulus_right,
    true_flow_left, true_flow_right,
    pred_flow_left, pred_flow_right,
    cell_x, cell_y, cell_z,
    output_path, frame_rate,
    mask_left=None, mask_right=None,
    stim_type="", flow_label="True Flow",
):
    """Render a 3x2 movie: stimulus / true flow / predicted flow for both eyes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus_left.shape[0]
    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    true_vx_r, true_vz_r, _, _, _ = prepare_eye_quiver(true_flow_right, cell_x, cell_y, cell_z, "right")
    true_vx_l, true_vz_l, _, _, _ = prepare_eye_quiver(true_flow_left, cell_x, cell_y, cell_z, "left")
    pred_vx_r, pred_vz_r, _, _, _ = prepare_eye_quiver(pred_flow_right, cell_x, cell_y, cell_z, "right")
    pred_vx_l, pred_vz_l, _, _, _ = prepare_eye_quiver(pred_flow_left, cell_x, cell_y, cell_z, "left")

    all_mag = np.hypot(
        np.concatenate([true_vx_r.ravel(), true_vx_l.ravel(),
                        pred_vx_r.ravel(), pred_vx_l.ravel()]),
        np.concatenate([true_vz_r.ravel(), true_vz_l.ravel(),
                        pred_vz_r.ravel(), pred_vz_l.ravel()]),
    )
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 16), dpi=100)
    ax_stim_l, ax_stim_r = axes[0]
    ax_true_l, ax_true_r = axes[1]
    ax_pred_l, ax_pred_r = axes[2]

    for ax in (ax_stim_l, ax_true_l, ax_pred_l):
        setup_eye_axis(ax, cell_x, cell_z, "left")
    for ax in (ax_stim_r, ax_true_r, ax_pred_r):
        setup_eye_axis(ax, cell_x, cell_z, "right")

    mask_l_0 = mask_left[0] if mask_left is not None else None
    mask_r_0 = mask_right[0] if mask_right is not None else None

    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], mask_l_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], mask_r_0, gray_cmap, stim_norm), s=90, edgecolors="none")
    ax_stim_l.set_title(f"Left Eye ({stim_type})")
    ax_stim_r.set_title(f"Right Eye ({stim_type})")

    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)

    q_true_l = init_eye_quiver(ax_true_l, cell_x, cell_z, true_vx_l[0], true_vz_l[0], max_mag, "left", **quiver_kw)
    q_true_r = init_eye_quiver(ax_true_r, cell_x, cell_z, true_vx_r[0], true_vz_r[0], max_mag, "right", **quiver_kw)
    ax_true_l.set_title(f"Left Eye — {flow_label}")
    ax_true_r.set_title(f"Right Eye — {flow_label}")

    q_pred_l = init_eye_quiver(ax_pred_l, cell_x, cell_z, pred_vx_l[0], pred_vz_l[0], max_mag, "left", **quiver_kw)
    q_pred_r = init_eye_quiver(ax_pred_r, cell_x, cell_z, pred_vx_r[0], pred_vz_r[0], max_mag, "right", **quiver_kw)
    ax_pred_l.set_title("Left Eye — Predicted Flow")
    ax_pred_r.set_title("Right Eye — Predicted Flow")

    plt.tight_layout()
    add_eye_colorwheel(fig)  # after tight_layout: the polar wheel is not a subplot

    def update(t):
        mask_l_t = mask_left[t] if mask_left is not None else None
        mask_r_t = mask_right[t] if mask_right is not None else None
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], mask_l_t, gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], mask_r_t, gray_cmap, stim_norm))
        update_eye_quiver(q_true_l, true_vx_l[t], true_vz_l[t], max_mag, "left")
        update_eye_quiver(q_true_r, true_vx_r[t], true_vz_r[t], max_mag, "right")
        update_eye_quiver(q_pred_l, pred_vx_l[t], pred_vz_l[t], max_mag, "left")
        update_eye_quiver(q_pred_r, pred_vx_r[t], pred_vz_r[t], max_mag, "right")
        ax_stim_l.set_title(f"Left Eye ({stim_type}) — frame {t + 1}/{num_frames}")
        ax_stim_r.set_title(f"Right Eye ({stim_type}) — frame {t + 1}/{num_frames}")

    write_movie(fig, update, num_frames, output_path, frame_rate)


def _make_ego_motion_movie(
    stimulus_left, stimulus_right,
    pred_flow_left, pred_flow_right,
    true_rot, pred_rot, true_trans, pred_trans,
    cell_x, cell_y, cell_z,
    in_fov_left, in_fov_right,
    output_path, frame_rate, model_dt,
    trail_sec=5.0, flow_label="predicted flow",
):
    """Movie: stimulus + predicted flow + rotation/translation rate plots."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    num_frames = stimulus_left.shape[0]
    trail_frames = int(round(trail_sec / model_dt))
    times = np.arange(num_frames) * model_dt

    gray_cmap = plt.cm.gray
    stim_min = min(np.nanmin(stimulus_left), np.nanmin(stimulus_right))
    stim_max = max(np.nanmax(stimulus_left), np.nanmax(stimulus_right))
    stim_norm = mcolors.Normalize(vmin=stim_min, vmax=stim_max)

    flow_vx_r, flow_vz_r, _, _, _ = prepare_eye_quiver(pred_flow_right, cell_x, cell_y, cell_z, "right")
    flow_vx_l, flow_vz_l, _, _, _ = prepare_eye_quiver(pred_flow_left, cell_x, cell_y, cell_z, "left")

    all_mag = np.hypot(
        np.concatenate([flow_vx_r.ravel(), flow_vx_l.ravel()]),
        np.concatenate([flow_vz_r.ravel(), flow_vz_l.ravel()]),
    )
    finite_mag = all_mag[np.isfinite(all_mag)]
    max_mag = (float(np.percentile(finite_mag, 99))
               if finite_mag.size > 0 and finite_mag.max() > 0 else 1.0)
    arrow_scale = 0.2 / max_mag if max_mag > 0 else 1.0

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.8), dpi=100,
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    ax_stim_l, ax_stim_r = axes[0]
    ax_flow_l, ax_flow_r = axes[1]
    ax_rot, ax_trans = axes[2]

    setup_eye_axis(ax_stim_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_stim_r, cell_x, cell_z, "right")
    sc_stim_l = ax_stim_l.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_left[0], get_fov(in_fov_left, 0), gray_cmap, stim_norm),
        s=90, edgecolors="none")
    sc_stim_r = ax_stim_r.scatter(cell_x, cell_z,
        c=masked_rgba(stimulus_right[0], get_fov(in_fov_right, 0), gray_cmap, stim_norm),
        s=90, edgecolors="none")
    ax_stim_l.set_title("Left Eye (stimulus)")
    ax_stim_r.set_title("Right Eye (stimulus)")

    setup_eye_axis(ax_flow_l, cell_x, cell_z, "left")
    setup_eye_axis(ax_flow_r, cell_x, cell_z, "right")
    quiver_kw = dict(angles="xy", scale_units="xy", scale=1 / arrow_scale, width=0.003)
    q_left = init_eye_quiver(ax_flow_l, cell_x, cell_z, flow_vx_l[0], flow_vz_l[0], max_mag, "left", **quiver_kw)
    q_right = init_eye_quiver(ax_flow_r, cell_x, cell_z, flow_vx_r[0], flow_vz_r[0], max_mag, "right", **quiver_kw)
    ax_flow_l.set_title(f"Left Eye ({flow_label})")
    ax_flow_r.set_title(f"Right Eye ({flow_label})")

    rot_labels = ["roll", "pitch", "yaw"]
    trans_labels = ["x", "y", "z"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    rot_true_lines, rot_pred_lines = [], []
    for label, color in zip(rot_labels, colors):
        lt, = ax_rot.plot([], [], color=color, linewidth=1.5, label=f"true {label}")
        lp, = ax_rot.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"pred {label}")
        rot_true_lines.append(lt)
        rot_pred_lines.append(lp)
    ax_rot.legend(loc="upper left", fontsize=7, ncol=2)
    ax_rot.set_ylabel("deg/s")
    ax_rot.set_xlabel("time (s)")
    ax_rot.set_title("Rotation Rate")
    all_rot = np.concatenate([true_rot, pred_rot], axis=0)
    ax_rot.set_ylim(*ylim_from_data(all_rot, min_margin=1.0))

    trans_true_lines, trans_pred_lines = [], []
    for label, color in zip(trans_labels, colors):
        lt, = ax_trans.plot([], [], color=color, linewidth=1.5, label=f"true {label}")
        lp, = ax_trans.plot([], [], color=color, linewidth=1.5, linestyle="--", label=f"pred {label}")
        trans_true_lines.append(lt)
        trans_pred_lines.append(lp)
    ax_trans.legend(loc="upper left", fontsize=7, ncol=2)
    ax_trans.set_ylabel("velocity")
    ax_trans.set_xlabel("time (s)")
    ax_trans.set_title("Translation Rate")
    all_trans = np.concatenate([true_trans, pred_trans], axis=0)
    ax_trans.set_ylim(*ylim_from_data(all_trans, min_margin=0.01))

    plt.tight_layout()
    add_eye_colorwheel(fig)  # after tight_layout: the polar wheel is not a subplot

    def update(t):
        sc_stim_l.set_facecolors(masked_rgba(stimulus_left[t], get_fov(in_fov_left, t), gray_cmap, stim_norm))
        sc_stim_r.set_facecolors(masked_rgba(stimulus_right[t], get_fov(in_fov_right, t), gray_cmap, stim_norm))
        update_eye_quiver(q_left, flow_vx_l[t], flow_vz_l[t], max_mag, "left")
        update_eye_quiver(q_right, flow_vx_r[t], flow_vz_r[t], max_mag, "right")

        time_s = t * model_dt
        ax_stim_l.set_title(f"Left Eye (t={time_s:.2f}s)")
        ax_stim_r.set_title(f"Right Eye (t={time_s:.2f}s)")

        t_start = max(0, t - trail_frames + 1)
        t_slice = slice(t_start, t + 1)
        t_times = times[t_slice]

        for i in range(3):
            rot_true_lines[i].set_data(t_times, true_rot[t_slice, i])
            rot_pred_lines[i].set_data(t_times, pred_rot[t_slice, i])
        ax_rot.set_xlim(t_times[0], t_times[0] + trail_sec)

        for i in range(3):
            trans_true_lines[i].set_data(t_times, true_trans[t_slice, i])
            trans_pred_lines[i].set_data(t_times, pred_trans[t_slice, i])
        ax_trans.set_xlim(t_times[0], t_times[0] + trail_sec)

    write_movie(fig, update, num_frames, output_path, frame_rate, log_every=100)


# ---------------------------------------------------------------------------
# Static figure helpers
# ---------------------------------------------------------------------------

def _save_magnitude_figure(true_mag_right, true_mag_left,
                           pred_mag_right, pred_mag_left,
                           model_dt, output_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    T = true_mag_right.shape[0]
    times = np.arange(T) * model_dt
    true_mag_mean = (np.nanmean(true_mag_right, axis=1) + np.nanmean(true_mag_left, axis=1)) / 2
    pred_mag_mean = (np.nanmean(pred_mag_right, axis=1) + np.nanmean(pred_mag_left, axis=1)) / 2

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, true_mag_mean, color="#e41a1c", linewidth=1.5, label="True magnitude (mean)")
    ax.plot(times, pred_mag_mean, color="#377eb8", linewidth=1.5, linestyle="--", label="Predicted magnitude (mean)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion magnitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Mean Motion Magnitude: True vs Predicted", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved magnitude figure -> {output_path}")


if __name__ == "__main__":
    main()
