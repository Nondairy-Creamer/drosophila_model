import warnings
import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
import torch

# Array conventions: the single-example make_* helpers return
#   stimulus: [time, points in space], optic flow: [time, points in space, 2];
# the batch create_* / get_* generators add a leading example axis:
#   stimulus: [examples, time, points in space]
#   optic flow: [examples, time, points in space, LR/UD optic flow]

from drosophila_model.config import get_data_folder, load_config
from drosophila_model.data_handling.sampling import (
    build_ommatidium_sampling_matrix,
    interpolate_to_model_dt,
)
from drosophila_model.data_handling import drone_utilities as du

# Full-sphere equirectangular panorama for expanding-dot: observer in uniform middle gray;
# every direction is a pixel.  φ ∈ [−π, π] (360° azimuth), θ ∈ [0, π] colatitude from +z
# (north pole) to −z (south pole).  W = 2H gives Δφ = Δθ in radians (equal angular step).
EXPANDING_DOT_PANO_HEIGHT = 512
EXPANDING_DOT_PANO_WIDTH = 1024
EXPANDING_DOT_PANO_PHI_LIM = np.array([np.pi, -np.pi])
EXPANDING_DOT_PANO_THETA_LIM = np.array([0.0, np.pi])

# World-space defaults for ``make_expanding_dot_pano`` (centimeters, seconds).
EXPANDING_DOT_RADIUS_CM = 5.0
EXPANDING_DOT_D0_RANGE_CM = (30.0, 90.0)
EXPANDING_DOT_SPEED_RANGE_CM_S = (15.0, 45.0)


def bake_fly_view_dataset(
    data_path: str | Path | None = None,
    out_path: str | Path | None = None,
    data_folder: str | Path | None = None,
):
    """Bake the raw fly-view pickle into a compact ``flyview.pt`` cache.

    Reads the raw fly-view pickle once and applies the two expensive,
    timing-independent transforms so the on-the-fly loader never has to touch
    the (multi-GB, ``float64``, depth-carrying) pickle again:

    * **Unit conversion**: both optic-flow components are converted from
      rad/frame to deg/s.
    * **Recalculated odometry**: per-clip rotation (deg/s) and translation
      (depth-units/s) are solved from the flow plus the rendered per-column
      ``depths`` via :func:`ego_motion.estimate_ego_motion_known_depth`.

    ``depths`` are consumed here and **not** stored. Invalid columns (NaN flow
    or |flow| > 1 rad/frame) are excluded from the odometry solve and then
    zeroed in the stored stimulus/flow. Everything is saved as ``float32``.

    Columns stay **block-packed**: ``[..., :n_eye]`` = right eye,
    ``[..., n_eye:]`` = left eye. Clips keep their native length and frame rate;
    ``drop_first_frame``, interpolation to the model ``dt`` and segmentation
    happen later in :func:`load_fly_view_dataset`.

    Args:
        data_path: Path to the raw pickle. When ``None``, resolves to
            ``<data_folder>/stimulus/raw/fib_forest_fly_view_dataset_fwd.pkl``.
        out_path: Destination ``.pt``. When ``None``, resolves to
            ``<data_folder>/stimulus/baked/flyview.pt``.
        data_folder: Data root, also used for the column viewing directions.
            When ``None``, uses ``[data].data_folder`` of the working-directory
            ``config.toml``.

    Returns:
        The path of the written ``flyview.pt``.
    """
    from drosophila_model.model_definition.ego_motion import (
        load_viewing_directions,
        estimate_ego_motion_known_depth,
    )

    if data_folder is None:
        data_folder = get_data_folder(load_config())
    stim_dir = Path(data_folder) / 'stimulus'
    if data_path is None:
        data_path = stim_dir / 'raw' / 'fib_forest_fly_view_dataset_fwd.pkl'
    if out_path is None:
        out_path = stim_dir / 'baked' / 'flyview.pt'
    in_path, out_path = Path(data_path), Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Fly-view pickle not found: {in_path.resolve()}")

    with open(in_path, 'rb') as f:
        d = pickle.load(f)

    fps = float(d['fps'])
    flow_units = d.get('flow_units', 'radians/frame')
    if flow_units != 'radians/frame':
        raise ValueError(
            f"Expected forward_flows in radians/frame; got flow_units={flow_units!r}"
        )
    # rad/frame -> deg/s (constant scale; temporal interpolation happens at load).
    flow_to_deg_s = fps * (180.0 / np.pi)

    n_cols = int(np.asarray(d['responses'][0]).shape[1])
    if n_cols % 2 != 0:
        raise ValueError(f"Expected an even column count (R|L); got {n_cols}")
    n_eye = n_cols // 2

    # Viewing directions ([right|left] block order, matching the column layout)
    # are needed to solve flow-based odometry.
    column_phi, column_theta = load_viewing_directions(data_folder)
    if column_phi.shape[0] != n_cols:
        raise ValueError(
            f"Viewing directions ({column_phi.shape[0]}) != flyview columns "
            f"({n_cols}); cannot solve flow-based odometry.")

    stim_clips, flow_clips, rot_clips, trans_clips = [], [], [], []
    for i in range(len(d['responses'])):
        resp = np.asarray(d['responses'][i], dtype=np.float64)
        flow = np.asarray(d['forward_flows'][i], dtype=np.float64)
        depth = np.asarray(d['depths'][i], dtype=np.float64)
        if resp.shape[0] != flow.shape[0]:
            raise ValueError(
                f"Clip {i}: response time {resp.shape[0]} != flow time {flow.shape[0]}")
        if resp.shape[0] == 0:
            continue

        bad = np.isnan(flow).any(axis=-1) | (np.hypot(flow[..., 0], flow[..., 1]) > 1.0)
        resp = np.nan_to_num(resp, nan=0.0)
        flow_deg_s = np.nan_to_num(flow, nan=0.0) * flow_to_deg_s

        # Ground-truth ego motion is derived from the flow itself plus the
        # rendered per-column depth, so the odometry targets are exactly
        # consistent with the flow the model sees. Solve in rad/s, then store
        # rotation in deg/s (matching the other odometry targets) and translation
        # in depth-units/s.
        flow_rad_s = flow_deg_s * (np.pi / 180.0)
        omega_rad, trans_units = estimate_ego_motion_known_depth(
            column_phi, column_theta,
            torch.from_numpy(flow_rad_s[..., 0]),
            torch.from_numpy(flow_rad_s[..., 1]),
            torch.from_numpy(depth),
            valid=torch.from_numpy(~bad),
        )
        omega = omega_rad.numpy() * (180.0 / np.pi)
        trans_rate = trans_units.numpy()

        # Zero invalid columns now (odometry already excluded them via ``valid``).
        resp[bad] = 0.0
        flow_deg_s = np.where(bad[..., np.newaxis], 0.0, flow_deg_s)

        stim_clips.append(torch.from_numpy(resp.astype(np.float32)))
        flow_clips.append(torch.from_numpy(flow_deg_s.astype(np.float32)))
        rot_clips.append(torch.from_numpy(omega.astype(np.float32)))
        trans_clips.append(torch.from_numpy(trans_rate.astype(np.float32)))

    if not stim_clips:
        raise RuntimeError("No fly-view clips produced (empty dataset?).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'fps': fps,
        'n_eye': n_eye,
        'stimulus': stim_clips,
        'optic_flow': flow_clips,
        'rotation_rate': rot_clips,
        'translation_rate': trans_clips,
    }, out_path)
    print(f"Baked 'flyview' ({len(stim_clips)} clips, n_eye={n_eye}) -> {out_path}")
    return out_path


def load_fly_view_dataset(
    model_dt: float,
    stim_time: float,
    *,
    two_eyes: bool,
    data_path: str | Path | None = None,
    drop_first_frame: bool = True,
):
    """
    Load the baked fly-view dataset (``<data_folder>/stimulus/baked/flyview.pt``).

    The flow (deg/s) and the depth-based odometry are precomputed by
    :func:`bake_fly_view_dataset`; this loader only handles the timing-dependent
    steps: drop the first frame, interpolate each whole clip to ``model_dt``,
    then cut it into non-overlapping segments of ``round(stim_time / model_dt)``
    steps (the length every other generator produces). Time is never
    stretched, so the deg/s flow and rate targets match the frames.

    Columns are **block-packed**: ``[..., :n_eye]`` = all right-eye samples,
    ``[..., n_eye:]`` = all left-eye samples (``n_eye = N // 2``).

    Args:
        model_dt: Model time step (seconds).
        stim_time: Duration of each segment (seconds). Clips shorter than one
            segment are skipped; a ``stim_time`` longer than the longest clip
            is capped to that clip's duration (with a warning).
        two_eyes: If True, return binocular columns ``[right | left]`` plus the
            baked ego-motion rates. If False, return right-eye columns only.
        data_path: Path to the baked ``flyview.pt``. When ``None``, resolves to
            ``<data_folder>/stimulus/baked/flyview.pt`` using ``[data].data_folder``
            of the working-directory ``config.toml``.
        drop_first_frame: If True, drop the first time index.

    Returns:
        If ``two_eyes`` is False:
            ``(stimulus, optic_flow, clip_ids)`` with stimulus/flow numpy
            arrays of shapes ``(batch, T, n_eye)`` and ``(batch, T, n_eye, 2)``.
        If ``two_eyes`` is True:
            ``(stimulus, optic_flow, rotation_rate, translation_rate, clip_ids)``
            with stimulus/flow ``(batch, T, 2*n_eye)`` / ``(batch, T, 2*n_eye, 2)``,
            ``rotation_rate`` ``(batch, T, 3)`` in deg/s and
            ``translation_rate`` ``(batch, T, 3)`` in depth-units per second.
        ``clip_ids`` is a ``(batch,)`` int array giving the source-clip index
        each segment was cut from, so callers can keep all segments of a clip
        on the same side of a train/test split.
    """
    if data_path is None:
        data_path = get_data_folder(load_config()) / 'stimulus' / 'baked' / 'flyview.pt'
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baked fly-view dataset not found: {path.resolve()}. "
            "Run `python -m drosophila_model.data_handling.process_raw_data flyview` "
            "to create it from fib_forest_fly_view_dataset_fwd.pkl.")

    d = torch.load(path, weights_only=False)
    stimulus_dt = 1.0 / float(d['fps'])
    n_eye = int(d['n_eye'])

    # Interpolate each whole clip to model_dt first, then segment, so every
    # segment spans exactly seg_len * model_dt seconds of the clip.
    # Per clip: [stimulus, optic_flow] (+ [rotation_rate, translation_rate]).
    clips = []
    for i in range(len(d['stimulus'])):
        arrays = [np.asarray(d['stimulus'][i]), np.asarray(d['optic_flow'][i])]
        if two_eyes:
            arrays += [np.asarray(d['rotation_rate'][i]),
                       np.asarray(d['translation_rate'][i])]
        else:
            arrays = [arrays[0][:, :n_eye], arrays[1][:, :n_eye]]
        if drop_first_frame:
            arrays = [a[1:] for a in arrays]
        if arrays[0].shape[0] < 2:
            continue
        arrays = [interpolate_to_model_dt(a[None], stimulus_dt, model_dt)[0] for a in arrays]
        clips.append((i, arrays))

    if not clips:
        raise RuntimeError("No rendered fly-view clips found (empty dataset?).")

    seg_len = max(1, int(round(stim_time / model_dt)))
    max_clip_len = max(arrays[0].shape[0] for _, arrays in clips)
    if seg_len > max_clip_len:
        warnings.warn(
            f"stim_time={stim_time}s ({seg_len} steps) exceeds the longest "
            f"fly-view clip ({max_clip_len} steps = {max_clip_len * model_dt:.2f}s). "
            f"Capping segments to {max_clip_len * model_dt:.2f}s."
        )
        seg_len = max_clip_len

    segments = [[] for _ in clips[0][1]]
    clip_id_list = []
    for i, arrays in clips:
        num_segments = arrays[0].shape[0] // seg_len
        if num_segments == 0:
            continue
        usable = num_segments * seg_len
        for out, a in zip(segments, arrays):
            out.append(a[:usable].reshape(num_segments, seg_len, *a.shape[1:]))
        clip_id_list.append(np.full(num_segments, i, dtype=np.int64))

    outputs = [np.concatenate(s, axis=0) for s in segments]
    clip_ids = np.concatenate(clip_id_list)
    return (*outputs, clip_ids)


def _stim_time_vector(dt, stim_time, pre_stim_time=0.0, post_stim_time=0.0):
    """Build a stimulus time vector (seconds) with the pre- and post-stimulus
    intervals clamped to hold the first/last stimulus sample.

    Returns ``(t, pre_idx, post_idx)`` where ``t`` is ``(num_time,)`` and the
    stimulus interval is ``t[pre_idx:post_idx]``.
    """
    num_time = int(round((pre_stim_time + stim_time + post_stim_time) / dt))
    t = np.arange(num_time) * dt
    pre_idx = int(round(pre_stim_time / dt))
    post_idx = int(round((pre_stim_time + stim_time) / dt))
    t[:pre_idx] = t[pre_idx]
    t[post_idx:] = t[post_idx - 1]
    return t, pre_idx, post_idx


def _contrast_scaling(sample_theta, wavelength_rad, filter_std_base=2.5 * np.pi / 180):
    """Per-column contrast attenuation from the ommatidia Gaussian filter.

    The acceptance-angle std (2.5 deg) widens toward the poles as
    ``1 / cos(theta - pi/2)``, damping the sine-wave amplitude there.
    """
    filter_std = filter_std_base / np.cos(sample_theta - np.pi / 2)
    return np.exp(-2 * np.pi**2 * filter_std**2 / wavelength_rad**2)


def _drift_flow(sample_phi, sample_theta, velocity_rad_s, axis):
    """Optic flow (deg/s) of a drifting grating moving at ``velocity_rad_s``.

    ``velocity_rad_s`` may be a scalar or a ``(T,)`` array. The flow is derived
    from the rigid ego-motion that would produce the same retinal slip on a
    virtual cylinder:

    * ``'lr'``: yaw rotation about z (``omega_z = velocity``), depth ``Z = 1``.
    * ``'ud'``: vertical translation (``T_z = -velocity``) at the cylinder wall
      ``Z = 1/sin(theta)`` (unit radius at the equator).

    Returns a ``(T, N, 2)`` array (``T = 1`` for scalar velocity) of
    ``(right, up)`` angular flow in degrees per second.
    """
    from drosophila_model.model_definition.ego_motion import generate_optic_flow

    v = np.atleast_1d(np.asarray(velocity_rad_s, dtype=np.float32))
    n_time = v.shape[0]
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)
    omega = torch.zeros(n_time, 3, dtype=torch.float32)
    T_vec = torch.zeros(n_time, 3, dtype=torch.float32)
    v_t = torch.tensor(v, dtype=torch.float32)
    if axis == 'lr':
        omega[:, 2] = v_t
        Z = 1.0
    elif axis == 'ud':
        T_vec[:, 2] = -v_t
        Z = 1.0 / torch.sin(theta_t)
    else:
        raise ValueError(f"axis must be 'lr' or 'ud', got {axis!r}")

    u_right, u_up = generate_optic_flow(phi_t, theta_t, omega, T_vec, Z)
    flow = np.stack([u_right.numpy() * (180.0 / np.pi),
                     u_up.numpy() * (180.0 / np.pi)], axis=-1)
    return flow.reshape(n_time, len(sample_phi), 2)


def _drift_ego(velocity_rad_s, axis, n_time):
    """Observer ego-motion that produces the same retinal slip as a drifting
    grating moving at ``velocity_rad_s`` (scalar or ``(T,)`` array).

    This is the ego-motion side of :func:`_drift_flow`, so the returned targets
    are exactly the ``(omega, T)`` used to synthesise the optic flow:

    * ``'lr'``: yaw rotation about z. ``rotation[:, 2] = velocity`` (deg/s),
      zero translation.
    * ``'ud'``: vertical translation. ``translation[:, 2] = -velocity`` (length
      per second on the unit-radius cylinder), zero rotation.

    Returns ``(rotation, translation)``, each ``(n_time, 3)``. Rotation is in
    deg/s (matching the other odometry targets); translation is linear velocity.
    """
    v = np.atleast_1d(np.asarray(velocity_rad_s, dtype=np.float64))
    if v.shape[0] == 1:
        v = np.full(n_time, v[0])
    elif v.shape[0] != n_time:
        raise ValueError(f"velocity length {v.shape[0]} != n_time {n_time}")

    rotation = np.zeros((n_time, 3))
    translation = np.zeros((n_time, 3))
    if axis == 'lr':
        rotation[:, 2] = v * (180.0 / np.pi)  # rad/s -> deg/s
    elif axis == 'ud':
        translation[:, 2] = -v
    else:
        raise ValueError(f"axis must be 'lr' or 'ud', got {axis!r}")
    return rotation, translation


def _make_drifting_sine(sample_phi, sample_theta, axis, contrast=1, wavelength=30,
                        temporal_frequency=4, phase_init=0, dt=0.01, stim_time=1.0,
                        pre_stim_time=0.0, post_stim_time=0.0):
    """Core drifting sine-wave generator shared by the LR and UD variants.

    ``axis='lr'`` drifts along azimuth (``phi``); ``axis='ud'`` drifts along the
    cartesian-vertical coordinate ``tan(theta - pi/2)`` so it moves at constant
    apparent speed on the virtual cylinder. See :func:`make_lr_sine_wave` /
    :func:`make_ud_sine_wave` for the public-facing argument descriptions.
    """
    wavelength_rad = wavelength * np.pi / 180
    t, pre_idx, post_idx = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)
    contrast_scaling_factor = _contrast_scaling(sample_theta, wavelength_rad)

    if axis == 'lr':
        spatial_coord = sample_phi / wavelength_rad
    elif axis == 'ud':
        # Vertical wavelength in cartesian space (cylinder is flat vertically).
        vert_wavelength = 2 * np.tan(wavelength_rad / 2)
        spatial_coord = np.tan(sample_theta - np.pi / 2) / vert_wavelength
    else:
        raise ValueError(f"axis must be 'lr' or 'ud', got {axis!r}")

    t_mesh, coord_mesh = np.meshgrid(t, spatial_coord, indexing='ij')
    sine_wave = contrast * np.sin(2 * np.pi * (coord_mesh + temporal_frequency * t_mesh) + phase_init)
    sine_wave *= contrast_scaling_factor[None, :]

    velocity_rad_s = temporal_frequency * wavelength_rad
    flow_cols = _drift_flow(sample_phi, sample_theta, velocity_rad_s, axis)[0]  # (N, 2)
    optic_flow = np.zeros((sine_wave.shape[0], sine_wave.shape[1], 2))
    optic_flow[pre_idx:post_idx] = flow_cols[None, :, :]

    sine_wave = (sine_wave + 1.0) / 2.0
    return sine_wave, optic_flow


def make_lr_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic left-right (azimuthal) drifting sine wave stimulus.

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
        temporal_frequency: temporal frequency (positive = rightward, negative = leftward)
        phase_init: initial phase offset
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time

    Returns:
        sine_wave: [time, points in space] array
        optic_flow: [time, points in space, 2] array (LR/UD flow)
    """
    return _make_drifting_sine(
        sample_phi, sample_theta, 'lr', contrast=contrast, wavelength=wavelength,
        temporal_frequency=temporal_frequency, phase_init=phase_init, dt=dt,
        stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)


def make_ud_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=4,
                      phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a basic up-down (vertical) drifting sine wave stimulus.

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
        temporal_frequency: temporal frequency (positive = upward, negative = downward)
        phase_init: initial phase offset
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time

    Returns:
        sine_wave: [time, points in space] array
        optic_flow: [time, points in space, 2] array (LR/UD flow)
    """
    return _make_drifting_sine(
        sample_phi, sample_theta, 'ud', contrast=contrast, wavelength=wavelength,
        temporal_frequency=temporal_frequency, phase_init=phase_init, dt=dt,
        stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)


def make_lr_moving_edge(sample_phi, sample_theta, velocity=30, edge_type='on',
                        dt=0.01, stim_time=1.0, pre_stim_time=0.0,
                        post_stim_time=0.0, edge_width=2.5):
    """Generate a left-right moving edge stimulus (a full 0 -> 1 step).

    Unlike :func:`make_lr_edge` (gray -> bright/dark), every column steps
    between 0 and 1 as the edge passes; the edge width is configurable.

    Args:
        sample_phi: azimuthal angles of ommatidia (radians).
        sample_theta: polar angles of ommatidia (radians).
        velocity: edge speed in degrees/second (positive = rightward, i.e.
            decreasing phi, as in :func:`make_lr_edge`).
        edge_type: ``'on'`` (dark-to-light transition as edge passes each
            point, i.e. a bright leading edge) or ``'off'`` (light-to-dark,
            dark leading edge).
        dt: time step (seconds).
        stim_time: stimulus duration (seconds).
        pre_stim_time: blank time before the edge enters the field.
        post_stim_time: blank time after stimulus ends.
        edge_width: base Gaussian smoothing sigma in degrees (scaled by
            theta to match ommatidium acceptance angle, same as sine wave).

    Returns:
        stimulus: ``(time, columns)`` array with values in [0, 1].
    """
    from scipy.special import erf

    velocity_rad = velocity * np.pi / 180
    filter_std_base = edge_width * np.pi / 180

    t_eff, pre_idx, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)
    t_eff = t_eff - t_eff[pre_idx]  # time since stimulus onset

    # Start edge well outside the visual field. Rightward motion is decreasing
    # phi (phi increases to the left).
    phi_span = sample_phi.max() - sample_phi.min()
    margin = 4 * filter_std_base + 0.1 * phi_span
    if velocity_rad >= 0:
        start_pos = sample_phi.max() + margin
    else:
        start_pos = sample_phi.min() - margin
    edge_pos = start_pos - velocity_rad * t_eff  # (time,)

    # Theta-dependent sigma (wider near poles, same convention as sine wave)
    sigma = filter_std_base / np.abs(np.cos(sample_theta - np.pi / 2))

    # Gaussian-smoothed step via error function.
    # "Has the edge passed this point?" depends on direction:
    #   rightward (v>0): passed when edge_pos < phi  → diff = +(phi - edge_pos)
    #   leftward  (v<0): passed when edge_pos > phi  → diff = -(phi - edge_pos)
    # Multiplying by sign(velocity) unifies both cases.
    direction = np.sign(velocity_rad) if velocity_rad != 0 else 1.0
    diff = direction * (sample_phi[None, :] - edge_pos[:, None])  # (time, columns)
    step = 0.5 * (1.0 + erf(diff / (sigma[None, :] * np.sqrt(2))))

    if edge_type.lower() in ('off', 'dark'):
        step = 1.0 - step

    return step


def make_counterphase_sine(sample_phi, sample_theta, contrast=1,
                           wavelength_phi=30, wavelength_theta=30,
                           temporal_frequency=4,
                           phase_phi=0, phase_theta=0, phase_t=0,
                           dt=0.01, stim_time=1.0,
                           pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a counterphase (standing wave) sine stimulus on a sphere.

    A spatial checkerboard pattern that flickers in place without translating.
    Mathematically equivalent to the sum of two gratings drifting in opposite
    directions at the same speed.

    stimulus = sin(2π t tf + phase_t) · sin(2π φ/λ_φ + phase_φ) · sin(2π θ/λ_θ + phase_θ)

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength_phi: spatial wavelength in phi direction (degrees)
        wavelength_theta: spatial wavelength in theta direction (degrees)
        temporal_frequency: flicker frequency (Hz)
        phase_phi: spatial phase offset in phi (radians)
        phase_theta: spatial phase offset in theta (radians)
        phase_t: temporal phase offset (radians)
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: [time, points in space] array with values in [0, 1]
        optic_flow: [time, points in space, 2] array (zeros — no net motion)
    """
    wavelength_phi_rad = wavelength_phi * np.pi / 180
    wavelength_theta_rad = wavelength_theta * np.pi / 180

    t, _, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)

    # Contrast attenuation from ommatidia Gaussian filtering (same as LR sine)
    contrast_scaling = _contrast_scaling(sample_theta, wavelength_phi_rad)

    spatial_phi = np.sin(2 * np.pi * sample_phi / wavelength_phi_rad + phase_phi)
    spatial_theta = np.sin(2 * np.pi * sample_theta / wavelength_theta_rad + phase_theta)
    spatial_pattern = spatial_phi * spatial_theta * contrast_scaling

    temporal = np.sin(2 * np.pi * temporal_frequency * t + phase_t)

    stimulus = contrast * temporal[:, None] * spatial_pattern[None, :]
    stimulus = (stimulus + 1.0) / 2.0

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def create_counterphase_sine(sample_phi, sample_theta, contrast=1,
                              temporal_frequencies=None,
                              wavelengths_phi=None, wavelengths_theta=None,
                              seed=None, dt=0.01, stim_time=1.0,
                              pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate counterphase sine stimuli for all combinations of temporal
    frequency, phi wavelength, and theta wavelength, with randomized
    spatiotemporal phase for each.

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        temporal_frequencies: list of temporal frequencies in Hz (default [2, 4, 8])
        wavelengths_phi: list of phi wavelengths in degrees (default [30, 90])
        wavelengths_theta: list of theta wavelengths in degrees (default [30, 90])
        seed: random seed for phase randomization
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: [examples, time, points in space] array with values in [0, 1]
        optic_flow: [examples, time, points in space, 2] array (zeros)
    """
    if temporal_frequencies is None:
        temporal_frequencies = [2, 4, 8]
    if wavelengths_phi is None:
        wavelengths_phi = [30, 90]
    if wavelengths_theta is None:
        wavelengths_theta = [30, 90]

    rng = np.random.default_rng(seed)

    stim_list = []
    flow_list = []

    for tf in temporal_frequencies:
        for wl_phi in wavelengths_phi:
            for wl_theta in wavelengths_theta:
                phase_phi = rng.uniform(0, 2 * np.pi)
                phase_theta = rng.uniform(0, 2 * np.pi)
                phase_t = np.pi / 2 if tf == 0 else 0.0

                s, f = make_counterphase_sine(
                    sample_phi, sample_theta, contrast=contrast,
                    wavelength_phi=wl_phi, wavelength_theta=wl_theta,
                    temporal_frequency=tf,
                    phase_phi=phase_phi, phase_theta=phase_theta,
                    phase_t=phase_t,
                    dt=dt, stim_time=stim_time,
                    pre_stim_time=pre_stim_time,
                    post_stim_time=post_stim_time,
                )
                stim_list.append(s[None, :, :])
                flow_list.append(f[None, :, :, :])

    stimulus = np.concatenate(stim_list, axis=0)
    optic_flow = np.concatenate(flow_list, axis=0)
    return stimulus, optic_flow


def make_lr_edge(sample_phi, sample_theta, velocity, edge_polarity='light',
                 dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a left-right moving edge stimulus on a gray background.

    The edge is a Heaviside step smoothed by the ommatidia Gaussian filter
    (2.5 deg std, scaled by 1/cos(theta-pi/2) like the sine-wave stimuli).
    A light (ON) edge transitions from gray (0.5) to bright (1.0) as it
    passes; a dark (OFF) edge transitions from gray (0.5) to dark (0.0).

    Args:
        sample_phi: azimuthal angles of columns (radians)
        sample_theta: polar angles of columns (radians)
        velocity: edge speed in degrees/s (positive = rightward)
        edge_polarity: 'light' or 'dark'
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus gray period (seconds)
        post_stim_time: post-stimulus frozen period (seconds)

    Returns:
        stimulus: (time, N) array with values in [0, 1]
        optic_flow: (time, N, 2) array of zeros -- a placeholder, not the
            edge's motion (edges are analysis-only probes, never baked).
    """
    from scipy.special import erf as sp_erf

    velocity_rad = velocity * np.pi / 180
    filter_std_base = 2.5 * np.pi / 180

    t, pre_stim_index, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)

    filter_std = filter_std_base / np.cos(sample_theta - np.pi / 2)  # (N,)

    margin = 5 * filter_std_base
    if velocity_rad >= 0:
        start_pos = np.max(sample_phi) + margin
    else:
        start_pos = np.min(sample_phi) - margin

    # Rightward motion = decreasing phi in the fly coordinate system
    edge_pos = start_pos - velocity_rad * (t - t[pre_stim_index])  # (T,)

    sign = 1.0 if velocity >= 0 else -1.0
    diff = sign * (sample_phi[None, :] - edge_pos[:, None])  # (T, N)
    H_smooth = 0.5 * (1.0 + sp_erf(diff / (filter_std[None, :] * np.sqrt(2))))

    if edge_polarity == 'light':
        stimulus = 0.5 + 0.5 * H_smooth
    elif edge_polarity == 'dark':
        stimulus = 0.5 - 0.5 * H_smooth
    else:
        raise ValueError(f"Unknown edge_polarity: {edge_polarity}. Use 'light' or 'dark'.")

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def make_ud_edge(sample_phi, sample_theta, velocity, edge_polarity='light',
                 dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate an up-down moving edge stimulus on a gray background.

    The edge sweeps the visual field via rotation around the x-axis (forward)
    at constant angular velocity.  Each column transitions once — from gray
    to bright (light) or gray to dark (dark) — and stays at the new value.
    The sweep speed varies with position: faster for lateral-looking columns
    (near ±y) and slower near the x-axis poles.

    Args:
        sample_phi: azimuthal angles of columns (radians)
        sample_theta: polar angles of columns (radians)
        velocity: edge speed in degrees/s (positive = upward)
        edge_polarity: 'light' or 'dark'
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus gray period (seconds)
        post_stim_time: post-stimulus frozen period (seconds)

    Returns:
        stimulus: (time, N) array with values in [0, 1]
        optic_flow: (time, N, 2) array of zeros -- a placeholder, not the
            edge's motion (edges are analysis-only probes, never baked).

    The sweep direction is only correct for right-eye columns (``phi < 0``);
    see ``_gen_edge`` for how two-eye stimuli are built.
    """
    from scipy.special import erf as sp_erf

    velocity_rad = velocity * np.pi / 180
    filter_std_base = 2.5 * np.pi / 180

    t, pre_stim_index, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)

    # Column positions in Cartesian (x = forward, y = left, z = up)
    col_y = np.sin(sample_theta) * np.sin(sample_phi)  # (N,)
    col_z = np.cos(sample_theta)                        # (N,)

    # Rotation angle around x-axis (roll).
    # The edge normal is n = Rx(alpha) · z_hat = (0, -sin(alpha), cos(alpha)).
    # For right-eye columns (phi < 0 → col_y < 0), alpha decreasing from
    # pi/2 sweeps the edge upward (bottom columns transition first).
    margin = 5 * filter_std_base
    if velocity_rad >= 0:  # upward: edge starts below FOV
        alpha_start = np.pi / 2 + margin
    else:  # downward: edge starts above FOV
        alpha_start = -np.pi / 2 - margin

    alpha = alpha_start - velocity_rad * (t - t[pre_stim_index])  # (T,)

    # Signed distance from the edge great circle.
    # d = dot(column, n) where n = Rx(alpha) · z_hat = (0, -sin(alpha), cos(alpha))
    # d > 0 means the column is on the +n side of the edge.
    d = (-col_y[None, :] * np.sin(alpha[:, None])
         + col_z[None, :] * np.cos(alpha[:, None]))  # (T, N)

    # Convention: "swept" (behind the edge) is the side that has already
    # transitioned.  For upward sweep the swept side is below (d < 0 at
    # alpha ~ pi/2), for downward it is above (d > 0 at alpha ~ -pi/2).
    sign = 1.0 if velocity >= 0 else -1.0
    swept = -sign * d  # positive = swept

    # Running max ensures each column stays at its peak transition value.
    # Without this, the great circle wraps around and a trailing edge
    # would reverse the transition.
    swept = np.maximum.accumulate(swept, axis=0)

    H_smooth = 0.5 * (1.0 + sp_erf(swept / (filter_std_base * np.sqrt(2))))

    if edge_polarity == 'light':
        stimulus = 0.5 + 0.5 * H_smooth
    elif edge_polarity == 'dark':
        stimulus = 0.5 - 0.5 * H_smooth
    else:
        raise ValueError(f"Unknown edge_polarity: {edge_polarity}. Use 'light' or 'dark'.")

    optic_flow = np.zeros((stimulus.shape[0], stimulus.shape[1], 2))
    return stimulus, optic_flow


def make_full_field_flicker(num_points, temporal_frequency, dt, stim_time,
                            pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a full-field flicker stimulus where luminance at every spatial
    point oscillates sinusoidally at a given temporal frequency.

    Args:
        num_points: number of spatial points (columns).
        temporal_frequency: flicker frequency in Hz.
        dt: time step in seconds.
        stim_time: stimulus duration in seconds.
        pre_stim_time: pre-stimulus hold at mean gray (seconds).
        post_stim_time: post-stimulus hold at final value (seconds).

    Returns:
        stimulus: (time, num_points) array with values in [0, 1].
        optic_flow: (time, num_points, 2) array of zeros.
    """
    t_eff, pre_idx, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)
    t_eff = t_eff - t_eff[pre_idx]
    num_time = len(t_eff)

    luminance = 0.5 + 0.5 * np.sin(2 * np.pi * temporal_frequency * t_eff)

    stimulus = np.broadcast_to(luminance[:, None], (num_time, num_points)).copy()
    optic_flow = np.zeros((num_time, num_points, 2))

    return stimulus, optic_flow


def create_edge_stimulus(sample_phi, sample_theta, velocities=None,
                         dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate edge stimuli for all four directions and both polarities.

    For each velocity, creates light and dark edges moving left, right,
    up, and down (4 directions x 2 polarities per velocity).

    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        velocities: list of edge speeds in deg/s (default [30, 60, 120, 240])
        dt: time step (seconds)
        stim_time: stimulus duration (seconds)
        pre_stim_time: pre-stimulus time (seconds)
        post_stim_time: post-stimulus time (seconds)

    Returns:
        stimulus: (examples, time, N) array with values in [0, 1]
        optic_flow: (examples, time, N, 2) array of placeholder zeros
    """
    if velocities is None:
        velocities = [30, 60, 120, 240]
    if not isinstance(velocities, (list, np.ndarray)):
        velocities = [velocities]

    blocks = ['right', 'left', 'up', 'down']
    stim_dict = {b: [] for b in blocks}
    flow_dict = {b: [] for b in blocks}

    for velocity in velocities:
        for polarity in ['light', 'dark']:
            edge_r, flow_r = make_lr_edge(
                sample_phi, sample_theta, velocity=velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['right'].append(edge_r[None, :, :])
            flow_dict['right'].append(flow_r[None, :, :, :])

            edge_l, flow_l = make_lr_edge(
                sample_phi, sample_theta, velocity=-velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['left'].append(edge_l[None, :, :])
            flow_dict['left'].append(flow_l[None, :, :, :])

            edge_u, flow_u = make_ud_edge(
                sample_phi, sample_theta, velocity=velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['up'].append(edge_u[None, :, :])
            flow_dict['up'].append(flow_u[None, :, :, :])

            edge_d, flow_d = make_ud_edge(
                sample_phi, sample_theta, velocity=-velocity,
                edge_polarity=polarity, dt=dt, stim_time=stim_time,
                pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            stim_dict['down'].append(edge_d[None, :, :])
            flow_dict['down'].append(flow_d[None, :, :, :])

    stimulus, optic_flow = consolidate_stim(stim_dict, flow_dict)
    return stimulus, optic_flow


def create_balanced_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, temporal_frequency=[4],
                              dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate balanced sine wave stimuli with multiple phases for left/right/up/down directions.
    This function calls make_lr_sine_wave and make_ud_sine_wave to create all variations.
    
    Args:
        sample_phi: azimuthal angles (radians)
        sample_theta: polar angles (radians)
        contrast: contrast of the sine wave
        wavelength: spatial wavelength (degrees)
        temporal_frequency: list of temporal frequencies to test
        dt: time step
        stim_time: stimulus duration
        pre_stim_time: pre-stimulus time
        post_stim_time: post-stimulus time
    
    Returns:
        stimulus: [examples, time, points in space]
        optic_flow: [examples, time, points in space, LR/UD optic flow]
        rotation: [examples, time, 3] observer angular velocity (deg/s); nonzero
            z-component for left/right examples, zero for up/down.
        translation: [examples, time, 3] observer linear velocity; nonzero
            z-component for up/down examples, zero for left/right.
        group_id: [examples] int array; the four phase/contrast variants of
            one (temporal frequency, direction) condition share an id.
    """
    if type(temporal_frequency) is not list:
        temporal_frequency = [temporal_frequency]

    temporal_frequency = [np.abs(tf) for tf in temporal_frequency]
    wavelength_rad = wavelength * np.pi / 180

    # (block name, drift axis, temporal-frequency sign)
    directions = [('right', 'lr', 1), ('left', 'lr', -1),
                  ('up', 'ud', 1), ('down', 'ud', -1)]
    sine_wave_dict = {b: [] for b, _, _ in directions}
    optic_flow_dict = {b: [] for b, _, _ in directions}
    rotation_dict = {b: [] for b, _, _ in directions}
    translation_dict = {b: [] for b, _, _ in directions}
    group_dict = {b: [] for b, _, _ in directions}

    # For each temporal frequency and direction, emit four examples: phase 0,
    # phase pi/2, and their contrast inversions (1 - wave). The inverted-contrast
    # copies reuse the same optic flow and ego-motion (inversion does not change
    # the motion). LR drift is observer yaw rotation; UD drift is observer
    # vertical translation (see _drift_ego / _drift_flow).
    condition = 0
    for tf in temporal_frequency:
        for block, axis, sign in directions:
            wave_0, flow_0 = _make_drifting_sine(
                sample_phi, sample_theta, axis, contrast=contrast, wavelength=wavelength,
                temporal_frequency=sign * tf, phase_init=0,
                dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            wave_90, flow_90 = _make_drifting_sine(
                sample_phi, sample_theta, axis, contrast=contrast, wavelength=wavelength,
                temporal_frequency=sign * tf, phase_init=np.pi / 2,
                dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
            rot, trans = _drift_ego(sign * tf * wavelength_rad, axis, wave_0.shape[0])
            sine_wave_dict[block] += [wave_0[None], wave_90[None],
                                      (1.0 - wave_0)[None], (1.0 - wave_90)[None]]
            optic_flow_dict[block] += [flow_0[None], flow_90[None],
                                       flow_0[None], flow_90[None]]
            rotation_dict[block] += [rot[None]] * 4
            translation_dict[block] += [trans[None]] * 4
            group_dict[block] += [condition] * 4
            condition += 1

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)
    rotation, translation = _consolidate_ego(rotation_dict, translation_dict)
    group_id = np.concatenate([np.asarray(group_dict[b]) for b in group_dict])

    return stimulus, optic_flow, rotation, translation, group_id

def create_random_tf_sine_wave(sample_phi, sample_theta, contrast=1, wavelength=30, velocity_amp=150.0, velocity_time_scale=0.15, seed=None,
                               dt=0.01, stim_time=1.0, num_repeats=5):
    """Sine gratings drifting with a smoothly varying random velocity.

    Returns ``(stimulus, optic_flow, rotation, translation, group_id)``: the
    usual ``[examples, time, ...]`` arrays plus an ``[examples]`` int array in
    which each example and its contrast-inverted copy share an id.
    """
    # generates what a point of light at sample_phi[i] and sample[theta_i] would look like at each time point
    # from the perspective of a spherical obsrever looking at a cylinder with a sine wave stimulus on it
    # the velocity of this sine wave is a smoothed gaussian random walk
    # positive / negative temporal frequency is right / left and up / down sine waves

    # Convert velocity (deg/s) to temporal frequency (Hz): degrees cancel
    tf_amp = velocity_amp / wavelength
    # Convert wavelength to radians for spatial calculations
    wavelength = wavelength * np.pi / 180

    # Create RNG from seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # --- generate the space time coordinates ---
    # double the number of stim because we want to symmetrize the temporal frequnecies
    num_repeats = num_repeats * 2
    time_per_stim = int(round(stim_time / dt))
    num_time = time_per_stim * num_repeats  # even, and divisible by num_repeats
    time_vect = np.arange(num_time) * dt
    num_cell = len(sample_phi)

    # --- generate the shifting temporal frequency ---
    tf = rng.standard_normal(num_time // 2)
    tf = np.concatenate((tf, -tf), axis=0)
    tf_filter = 1 / (np.sqrt(2 * np.pi * velocity_time_scale**2)) * np.exp(-(time_vect-time_vect.mean())**2 / (2 * velocity_time_scale**2)) * dt
    tf_filter /= tf_filter.sum()
    tf_filter = np.fft.ifftshift(tf_filter)  # shift for our circular convolution
    tf = np.fft.ifft(np.fft.fft(tf) * np.fft.fft(tf_filter)).real
    # rescale the tf after smoothing to get the correct temporal frequency
    expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / velocity_time_scale)
    tf *= tf_amp / expected_std

    # smooth the sine wave according to its location in theta (ommatidia Gaussian
    # filtering with std 2.5 deg, widening toward the poles): the amplitude at the
    # top / bottom of the cylinder gets damped.
    contrast_scaling_factor = _contrast_scaling(sample_theta, wavelength)

    # length scales for vertical sine waves (since the cylinder is flat vertically)
    vert_wavelength = 2 * np.tan(wavelength / 2) # length of sine wave in cartesian space
    theta_to_length = np.tan(sample_theta - np.pi / 2)

    blocks = ['right_left', 'up_down']
    sine_wave_dict = {i: [] for i in blocks}
    optic_flow_dict = {i: [] for i in blocks}
    rotation_dict = {i: [] for i in blocks}
    translation_dict = {i: [] for i in blocks}

    def make_lr_sine_wave(phase_init=0):
        phase = phase_init
        sine_wave = np.zeros((num_time, num_cell))

        for ti in range(time_vect.shape[0]):
            phase += tf[ti] * dt
            sine_wave[ti, :] = contrast * np.sin(2 * np.pi * (sample_phi / wavelength + phase))

        return sine_wave

    def make_ud_sine_wave(phase_init=0):
        phase = phase_init
        sine_wave = np.zeros((num_time, num_cell))

        for ti in range(time_vect.shape[0]):
            phase += tf[ti] * dt
            sine_wave[ti, :] = contrast * np.sin(2 * np.pi * (theta_to_length / vert_wavelength + phase))

        return sine_wave

    # -- right/left sine wave --
    sine_wave_right_left = make_lr_sine_wave(phase_init=0)
    sine_wave_right_left_shifted = make_lr_sine_wave(phase_init=np.pi/2)
    sine_wave_right_left *= contrast_scaling_factor[None, :]
    sine_wave_right_left_shifted *= contrast_scaling_factor[None, :]
    sine_wave_dict['right_left'].append(sine_wave_right_left.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append(sine_wave_right_left_shifted.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append((-sine_wave_right_left).reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['right_left'].append((-sine_wave_right_left_shifted).reshape(num_repeats, time_per_stim, num_cell))

    # Optic flow for LR: rotation about the z-axis in the cylinder.
    velocity_rad_s = tf * wavelength  # (num_time,) in rad/s
    lr_flow = _drift_flow(sample_phi, sample_theta, velocity_rad_s, 'lr')
    optic_flow_mat = lr_flow.reshape(num_repeats, time_per_stim, num_cell, 2)

    optic_flow_dict['right_left'] += [optic_flow_mat] * 4

    # Ego-motion for LR: time-varying observer yaw rotation.
    lr_rot, lr_trans = _drift_ego(velocity_rad_s, 'lr', num_time)
    rotation_dict['right_left'] += [lr_rot.reshape(num_repeats, time_per_stim, 3)] * 4
    translation_dict['right_left'] += [lr_trans.reshape(num_repeats, time_per_stim, 3)] * 4

    # -- updown sine wave -- 
    sine_wave_up_down = make_ud_sine_wave(phase_init=0)
    sine_wave_up_down_shifted = make_ud_sine_wave(phase_init=np.pi/2)
    sine_wave_up_down *= contrast_scaling_factor[None, :]
    sine_wave_up_down_shifted *= contrast_scaling_factor[None, :]
    sine_wave_dict['up_down'].append(sine_wave_up_down.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append(sine_wave_up_down_shifted.reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down).reshape(num_repeats, time_per_stim, num_cell))
    sine_wave_dict['up_down'].append((-sine_wave_up_down_shifted).reshape(num_repeats, time_per_stim, num_cell))

    # Optic flow for UD: vertical translation on the cylinder wall (Z = 1/sin(theta)).
    ud_flow = _drift_flow(sample_phi, sample_theta, velocity_rad_s, 'ud')
    optic_flow_mat = ud_flow.reshape(num_repeats, time_per_stim, num_cell, 2)
    optic_flow_dict['up_down'] += [optic_flow_mat] * 4

    # Ego-motion for UD: time-varying observer vertical translation.
    ud_rot, ud_trans = _drift_ego(velocity_rad_s, 'ud', num_time)
    rotation_dict['up_down'] += [ud_rot.reshape(num_repeats, time_per_stim, 3)] * 4
    translation_dict['up_down'] += [ud_trans.reshape(num_repeats, time_per_stim, 3)] * 4

    stimulus, optic_flow = consolidate_stim(sine_wave_dict, optic_flow_dict)
    stimulus = (stimulus + 1.0) / 2.0
    rotation, translation = _consolidate_ego(rotation_dict, translation_dict)

    # Each block holds [wave, shifted, -wave, -shifted] x num_repeats; the
    # contrast-inverted copies (-wave, -shifted) share their original's group.
    repeat = np.arange(num_repeats)
    group_id = np.concatenate([
        b * 2 * num_repeats + np.concatenate([repeat, repeat + num_repeats] * 2)
        for b in range(len(blocks))
    ])

    return stimulus, optic_flow, rotation, translation, group_id


def create_directional_sine_wave(sample_phi, sample_theta, direction, contrast=1, wavelength=30, temporal_frequency=4,
                                 phase_init=0, dt=0.01, stim_time=1.0, pre_stim_time=0.0, post_stim_time=0.0):
    """
    Generate a single-direction sine wave stimulus.

    Returns ``(stimulus, optic_flow, rotation, translation)`` with leading
    example axis of size 1. ``rotation`` (deg/s) is nonzero for left/right
    (observer yaw); ``translation`` is nonzero for up/down (observer vertical
    translation). See :func:`_drift_ego`.
    """
    direction = direction.lower()
    if direction in ["right", "left"]:
        axis = 'lr'
        tf = temporal_frequency if direction == "right" else -temporal_frequency
    elif direction in ["up", "down"]:
        axis = 'ud'
        tf = temporal_frequency if direction == "up" else -temporal_frequency
    else:
        raise ValueError(f"Unsupported direction for sine wave: {direction}")

    sine_wave, optic_flow = _make_drifting_sine(
        sample_phi, sample_theta, axis, contrast=contrast, wavelength=wavelength,
        temporal_frequency=tf, phase_init=phase_init,
        dt=dt, stim_time=stim_time, pre_stim_time=pre_stim_time, post_stim_time=post_stim_time)
    rotation, translation = _drift_ego(tf * wavelength * np.pi / 180, axis, sine_wave.shape[0])

    stimulus = sine_wave[None, :, :]
    optic_flow = optic_flow[None, :, :, :]
    return stimulus, optic_flow, rotation[None], translation[None]


def _random_rotation_matrix(rng):
    """Return a uniformly random 3x3 rotation matrix (Arvo 1992, fast uniform
    sampling of SO(3) via QR decomposition of a random matrix)."""
    z = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(z)
    # Ensure a proper rotation (det = +1) by flipping signs of columns
    # whose corresponding diagonal entry of r is negative.
    d = np.diag(r)
    q *= np.sign(d)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _load_all_panos(pt_path=None, *, data_folder=None):
    """Load all panoramic images from a single .pt file.

    The .pt file contains a (N, H, W) float32 tensor of raw panoramic images
    (created by consolidating the individual .mat files).

    Args:
        pt_path: Path to ``all_panoramic.pt``. When ``None``, resolves to
            ``<data_folder>/stimulus/baked/all_panoramic.pt``.
        data_folder: Data root used when ``pt_path`` is ``None``; defaults to
            ``[data].data_folder`` of the working-directory ``config.toml``.

    Returns:
        panos: (N, H, W) numpy float64 array of raw panoramic images.
    """
    if pt_path is None:
        if data_folder is None:
            data_folder = get_data_folder(load_config())
        pt_path = Path(data_folder) / 'stimulus' / 'baked' / 'all_panoramic.pt'
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Consolidated panoramic file not found at {pt_path.resolve()}. "
            "Create it with `python -m drosophila_model.data_handling.process_raw_data "
            "random_pano`."
        )
    panos = torch.load(pt_path, weights_only=True)
    return panos.numpy().astype(np.float64)


def _filter_pano(pano, filter_std_rad=2.5 * np.pi / 180):
    """Apply Gaussian low-pass filter to a panoramic image for anti-aliasing.

    Blurs the raw panorama with a Gaussian kernel (ommatidia optical filtering)
    so the image can be safely sampled at the sparse column grid without
    aliasing.  Output is normalized to [0, 1].

    Args:
        pano: (H, W) raw panoramic image array.
        filter_std_rad: Gaussian sigma in radians (default 2.5 deg).

    Returns:
        pano_out: (H, W) float64 array with values in [0, 1].
        pano_y_lim: (2,) theta limits of the panorama.
        pano_x_lim: (2,) phi limits of the panorama.
    """
    from scipy.ndimage import gaussian_filter

    pano_y_lim = -np.array([47, -58]) * np.pi / 180 + np.pi / 2  # theta range
    pano_x_lim = np.array([np.pi, -np.pi])  # phi range

    pano_height, pano_width = pano.shape
    y_pixels_per_rad = pano_height / (pano_y_lim[1] - pano_y_lim[0])
    x_pixels_per_rad = pano_width / (pano_x_lim[0] - pano_x_lim[1])
    sigma_y = filter_std_rad * y_pixels_per_rad
    sigma_x = filter_std_rad * x_pixels_per_rad

    pano_out = gaussian_filter(
        pano.astype(np.float64), sigma=[sigma_y, sigma_x], mode=['nearest', 'wrap']
    )

    pano_min = pano_out.min()
    pano_max = pano_out.max()
    if pano_max > pano_min:
        pano_out = (pano_out - pano_min) / (pano_max - pano_min)
    else:
        pano_out = np.zeros_like(pano_out)

    return pano_out, pano_y_lim, pano_x_lim


def _rotation_increment(omega, dt):
    """Rotation matrix for turning at angular velocity ``omega`` ((3,), rad/s)
    for ``dt`` seconds (Rodrigues' formula)."""
    angle = np.linalg.norm(omega) * dt
    if angle <= 1e-12:
        return np.eye(3)
    ax = omega / np.linalg.norm(omega)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _sample_pano(pano_filtered, xyz, pano_y_lim, pano_x_lim):
    """Bilinearly sample a filtered panorama along unit viewing directions.

    ``xyz`` is ``(..., 3)`` in the panorama (world) frame, +x forward /
    +y left / +z up. Returns ``(stim, out_of_bounds)``, both shaped like
    ``xyz[..., 0]``; directions above or below the panorama are marked
    out of bounds and get the panorama mean.
    """
    from scipy.ndimage import map_coordinates

    pano_height, pano_width = pano_filtered.shape
    theta = np.arccos(np.clip(xyz[..., 2], -1, 1))
    phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    y_indices = (theta - pano_y_lim[0]) / (pano_y_lim[1] - pano_y_lim[0]) * (pano_height - 1)
    x_indices = (pano_x_lim[0] - phi) / (pano_x_lim[0] - pano_x_lim[1]) * pano_width
    x_indices = x_indices % pano_width  # wrap horizontally

    out_of_bounds = (y_indices < 0) | (y_indices > pano_height - 1)
    y_indices = np.clip(y_indices, 0, pano_height - 1)

    stim = map_coordinates(
        pano_filtered, [y_indices.ravel(), x_indices.ravel()],
        order=1, mode='nearest'
    ).reshape(y_indices.shape)
    stim[out_of_bounds] = pano_filtered.mean()
    return stim, out_of_bounds


def make_expanding_dot_pano(
    sample_phi,
    sample_theta,
    object_polarity,
    dt=0.01,
    stim_time=1.0,
    pre_stim_time=0.0,
    post_stim_time=0.0,
    *,
    physical_radius=None,
    approach_speed=None,
    initial_distance=None,
    center_phi=None,
    center_theta=None,
    seed=None,
    filter_std=2.5,
    ommatidium_sampling_matrix=None,
):
    """
    Gray-field (0.5) **full-sphere** equirectangular panorama: all directions are
    defined (observer in a uniform gray void).  Resolution is
    ``(EXPANDING_DOT_PANO_HEIGHT, EXPANDING_DOT_PANO_WIDTH)`` with uniform angular
    step in azimuth and colatitude over ``2π × π`` rad (full 360° by 180°).
    Default dot direction is **uniform on the sphere** (not uniform in colatitude).
    Angular radius follows ``alpha = atan(R / d(t))`` with ``d(t) = d0 - v t``
    (constant ``v``).  Default world units are **centimeters** and **seconds**:
    ``R = 5`` cm, ``d0`` uniform on ``[30, 90]`` cm, ``v`` uniform on ``[15, 45]``
    cm/s (override with ``physical_radius`` / ``initial_distance`` /
    ``approach_speed``).

    Column samples use :func:`build_ommatidium_sampling_matrix` and ``W @
    pano.ravel()`` each frame (same as the drone samplers), not bilinear
    ``map_coordinates`` like :func:`get_random_velocity_pano`.  Here ``W`` is
    shape ``(N, H*W)`` sparse (one row per ommatidium, one column per pano pixel) —
    not the same shape as the ``(H, W)`` image.  That matvec each timestep is why
    this stimulus can be heavier than :func:`get_random_velocity_pano`, which only
    **interpolates N probe locations** per frame after a rotation, even though the
    scene content (a disk) is simpler than a full photo panorama.

    Args:
        sample_phi: (N,) azimuth (radians).
        sample_theta: (N,) polar angle (radians).
        object_polarity: ``'light'`` (dot brighter than gray) or ``'dark'``.
        dt, stim_time, pre_stim_time, post_stim_time: timing (seconds).
        physical_radius: object radius ``R`` (cm by default; must match ``d0``).
        approach_speed: closing speed ``v`` in cm/s (default uniform 15–45). If
            None, drawn at random.
        initial_distance: starting distance ``d0`` in cm (default uniform 30–90).
            If None, drawn at random.
        center_phi, center_theta: dot center on the sphere (same spherical convention
            as ``sample_*``). If both None, direction is drawn uniformly on ``S^2``.
        seed: RNG seed for random draws.
        filter_std: ommatidium Gaussian std in degrees (default 2.5); ignored if
            ``ommatidium_sampling_matrix`` is provided.
        ommatidium_sampling_matrix: optional precomputed ``W`` from
            :func:`build_ommatidium_sampling_matrix` (same geometry for every trial
            in a batch).

    Returns:
        stimulus: (T, N) in [0, 1].
        optic_flow: (T, N, 2) zeros (no self-motion).
    """
    if object_polarity not in ('light', 'dark'):
        raise ValueError("object_polarity must be 'light' or 'dark'")

    rng = np.random.default_rng(seed)

    pano_phi_lim = EXPANDING_DOT_PANO_PHI_LIM
    pano_theta_lim = EXPANDING_DOT_PANO_THETA_LIM
    H, W = EXPANDING_DOT_PANO_HEIGHT, EXPANDING_DOT_PANO_WIDTH

    if physical_radius is None:
        R = float(EXPANDING_DOT_RADIUS_CM)
    else:
        R = float(physical_radius)
    if R <= 0:
        raise ValueError("physical_radius must be positive")

    if approach_speed is None:
        lo, hi = EXPANDING_DOT_SPEED_RANGE_CM_S
        v = float(rng.uniform(lo, hi))
    else:
        v = float(approach_speed)
    if v <= 0:
        raise ValueError("approach_speed must be positive")

    if initial_distance is None:
        lo, hi = EXPANDING_DOT_D0_RANGE_CM
        d0 = float(rng.uniform(lo, hi))
    else:
        d0 = float(initial_distance)
    if d0 <= 0:
        raise ValueError("initial_distance must be positive")

    if center_phi is None and center_theta is None:
        u_dir, v_dir = rng.random(), rng.random()
        theta0 = float(np.arccos(1.0 - 2.0 * u_dir))
        phi0 = float(2.0 * np.pi * v_dir - np.pi)
    else:
        if center_phi is None:
            phi0 = float(rng.uniform(-np.pi, np.pi))
        else:
            phi0 = float(center_phi)
        if center_theta is None:
            theta0 = float(np.arccos(1.0 - 2.0 * rng.random()))
        else:
            theta0 = float(center_theta)

    t, pre_stim_index, _ = _stim_time_vector(dt, stim_time, pre_stim_time, post_stim_time)
    num_time = len(t)
    t_stim = t - t[pre_stim_index]

    d = np.maximum(d0 - v * t_stim, 1e-9)
    alpha = np.arctan(R / d)

    # Match ``build_ommatidium_sampling_matrix`` pixel axes (inclusive endpoints).
    theta_pixels = np.linspace(pano_theta_lim[0], pano_theta_lim[1], H)
    phi_pixels = np.linspace(pano_phi_lim[0], pano_phi_lim[1], W)
    theta_grid, phi_grid = np.meshgrid(theta_pixels, phi_pixels, indexing='ij')
    cos_g = (
        np.sin(theta_grid) * np.sin(theta0) * np.cos(phi_grid - phi0)
        + np.cos(theta_grid) * np.cos(theta0)
    )
    cos_g = np.clip(cos_g, -1.0, 1.0)
    g_grid = np.arccos(cos_g)

    if object_polarity == 'light':
        inside_val = 1.0
    else:
        inside_val = 0.0

    num_cells = len(sample_phi)
    stimulus = np.full((num_time, num_cells), 0.5, dtype=np.float64)

    if ommatidium_sampling_matrix is not None:
        W_eye = ommatidium_sampling_matrix
    else:
        W_eye = build_ommatidium_sampling_matrix(
            (H, W),
            pano_phi_lim,
            pano_theta_lim,
            sample_phi,
            sample_theta,
            filter_std=filter_std,
        )

    # Per-frame pano construction (avoids the (T, H, W) float64 allocation).
    # alpha is monotone non-decreasing (frozen pre/post, growing during stim),
    # so each frame only flips additional pixels from gray to inside_val.
    pano_frame = np.full((H, W), 0.5, dtype=np.float64)
    prev_mask = np.zeros_like(g_grid, dtype=bool)
    for ti in range(num_time):
        mask = g_grid <= alpha[ti]
        newly_inside = mask & (~prev_mask)
        if newly_inside.any():
            pano_frame[newly_inside] = inside_val
        prev_mask = mask

        stimulus[ti] = np.asarray(
            W_eye @ pano_frame.ravel(order="C"), dtype=np.float64
        ).ravel()

    optic_flow = np.zeros((num_time, num_cells, 2), dtype=np.float64)
    return stimulus, optic_flow


def create_expanding_dot_stimulus(
    sample_phi,
    sample_theta,
    *,
    dt,
    stim_time,
    pre_stim_time=0.0,
    post_stim_time=0.0,
    seed=None,
    num_light=50,
    num_dark=50,
    filter_std=2.5,
):
    """
    Batch of light- and dark-polarity expanding-dot trials (see
    :func:`make_expanding_dot_pano`).

    The ommatidium matrix depends only on retina geometry (phi/theta columns) and
    pano shape — not on dot position or polarity — so it is built **once** here and
    reused for every trial (see ``ommatidium_sampling_matrix`` in
    :func:`make_expanding_dot_pano`).

    Returns:
        stimulus: (num_light + num_dark, T, N) in [0, 1].
        optic_flow: (num_light + num_dark, T, N, 2) zeros.
    """
    H, W_img = EXPANDING_DOT_PANO_HEIGHT, EXPANDING_DOT_PANO_WIDTH
    W_eye = build_ommatidium_sampling_matrix(
        (H, W_img),
        EXPANDING_DOT_PANO_PHI_LIM,
        EXPANDING_DOT_PANO_THETA_LIM,
        sample_phi,
        sample_theta,
        filter_std=filter_std,
    )

    master_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    stim_blocks = []
    flow_blocks = []
    for polarity, count in (('light', num_light), ('dark', num_dark)):
        for _ in range(count):
            trial_seed = int(master_rng.integers(0, 2**31 - 1))
            s, f = make_expanding_dot_pano(
                sample_phi,
                sample_theta,
                polarity,
                dt=dt,
                stim_time=stim_time,
                pre_stim_time=pre_stim_time,
                post_stim_time=post_stim_time,
                seed=trial_seed,
                ommatidium_sampling_matrix=W_eye,
            )
            stim_blocks.append(s[None, ...])
            flow_blocks.append(f[None, ...])
    return np.concatenate(stim_blocks, axis=0), np.concatenate(flow_blocks, axis=0)


def trim_expanding_dot_config_for_max_index(data_config, max_index: int) -> None:
    """
    Reduce ``expanding_dot.num_light`` / ``expanding_dot.num_dark`` so the batch
    is the smallest that still includes indices ``0 .. max_index`` (light trials
    first, then dark). Mutates ``data_config['expanding_dot']`` in place.

    Used by movie scripts that only need a few examples (e.g. ``--stim-index 0``)
    without paying for the full default 50+50 batch.
    """
    if max_index < 0:
        return
    dot_params = data_config['expanding_dot']
    nl = int(dot_params['num_light'])
    nd = int(dot_params['num_dark'])
    if max_index < nl:
        dot_params['num_light'] = max_index + 1
        dot_params['num_dark'] = 0
    else:
        dot_params['num_light'] = nl
        dot_params['num_dark'] = min(nd, max_index - nl + 1)


def get_random_velocity_pano(sample_phi, sample_theta, velocity_amp=180, velocity_time_scale=0.25, seed=None,
                             dt=0.01, stim_time=1.0, num_repeats=5, data_folder=None,
                             pano_indices=None):
    """
    Generate panoramic stimuli with smoothly varying rotational velocity AND
    smoothly varying rotation axis.

    Loads all panoramic images from ``<data_folder>/stimulus/baked/all_panoramic.pt``, applies
    2D Gaussian spatial filtering (std=2.5 deg), then for **each** panorama
    simulates pure rotation with an independent temporally-smoothed random
    angular velocity vector (omega_x, omega_y, omega_z).  Each of the three
    components is an independent smoothed Gaussian random walk, so both the
    magnitude and direction of rotation evolve smoothly over time.  At each
    time step the viewing directions are incrementally rotated and the filtered
    panorama is sampled.  Optic flow is computed analytically (linear in omega).

    Coordinate convention: +x forward, +y left, +z up.

    Args:
        sample_phi: (N,) azimuthal angles of viewing directions (radians)
        sample_theta: (N,) polar angles of viewing directions (radians)
        velocity_amp: RMS amplitude of rotational speed (deg/s)
        velocity_time_scale: temporal smoothing scale for velocity (seconds)
        seed: random seed
        dt: time step (seconds)
        stim_time: duration per repeat (seconds)
        num_repeats: number of repeats *per panorama* (will be doubled for
            symmetrization)
        data_folder: data root holding ``stimulus/baked/all_panoramic.pt``;
            defaults to ``[data].data_folder`` of the working-directory
            ``config.toml``.
        pano_indices: optional 1D array of indices into ``all_panoramic.pt``;
            default uses every panorama. The random draws are made pano by
            pano, so ``pano_indices=range(k)`` reproduces the first ``k``
            panoramas of the default output.

    Returns:
        stimulus: [examples, time, points in space] — pixel values in [0, 1]
        optic_flow: [examples, time, points in space, 2] — (d_right, d_up) in deg/s
        rotation_vectors: [examples, time, 3] — body-frame angular velocity (rad/s)
        oob_mask: [examples, time, points in space] — True where viewing direction
            falls outside panorama bounds (above/below image).
        Examples are panorama-major: the ``2 * num_repeats`` repeats of each
        panorama are consecutive.
    """
    from drosophila_model.model_definition.ego_motion import generate_optic_flow

    # Create RNG from seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # --- load all panoramic images ---
    raw_panos = _load_all_panos(data_folder=data_folder)
    if pano_indices is not None:
        raw_panos = raw_panos[np.asarray(pano_indices, dtype=int)]

    pano_images = []
    for i in range(raw_panos.shape[0]):
        pano_filtered, pano_y_lim, pano_x_lim = _filter_pano(raw_panos[i])
        pano_images.append(pano_filtered)

    # --- time / velocity parameters (shared across panoramas) ---
    num_repeats_total = num_repeats * 2  # double for symmetrization
    time_per_stim = int(round(stim_time / dt))
    num_time = time_per_stim * num_repeats_total
    time_vect = np.arange(num_time) * dt

    # Gaussian smoothing filter (circular convolution)
    vel_filter = (1 / np.sqrt(2 * np.pi * velocity_time_scale**2)
                  * np.exp(-(time_vect - time_vect.mean())**2 / (2 * velocity_time_scale**2)) * dt)
    vel_filter /= vel_filter.sum()
    vel_filter = np.fft.ifftshift(vel_filter)

    expected_std = 1 / np.sqrt(2 * np.sqrt(np.pi)) * np.sqrt(dt / velocity_time_scale)
    velocity_amp_rad = velocity_amp * np.pi / 180  # convert deg/s to rad/s

    # Scale per-component amplitude so the total angular speed has the desired RMS
    # E[|omega|^2] = 3 * sigma_comp^2,  so sigma_comp = velocity_amp / sqrt(3)
    component_amp = velocity_amp_rad / np.sqrt(3)

    # --- convert sample viewing directions to Cartesian ---
    sample_xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta)
    ], axis=1)  # (N, 3)
    num_cells = len(sample_phi)

    T_zero = torch.tensor([0.0, 0.0, 0.0])
    phi_t = torch.tensor(sample_phi, dtype=torch.float32)
    theta_t = torch.tensor(sample_theta, dtype=torch.float32)

    stimulus_list = []
    optic_flow_list = []
    omega_body_list = []
    oob_list = []

    for pano_filtered in pano_images:
        # --- generate 3 independent smoothed angular velocity components ---
        omega = np.zeros((3, num_time))
        for dim in range(3):
            v = rng.standard_normal(num_time // 2)
            v = np.concatenate((v, -v))  # symmetrize: second half is negated
            v = np.fft.ifft(np.fft.fft(v) * np.fft.fft(vel_filter)).real
            v *= component_amp / expected_std
            omega[dim] = v

        # Reshape into per-repeat segments: (3, num_repeats_total, time_per_stim)
        omega_per_repeat = omega.reshape(3, num_repeats_total, time_per_stim)

        for ri in range(num_repeats_total):
            omega_seg = omega_per_repeat[:, ri, :]  # (3, time_per_stim)

            # --- compose incremental rotations (Rodrigues' formula) ---
            # The update R = R_inc @ R is left-multiplication, so omega_seg is
            # in the world frame.  For optic-flow we need the body-frame
            # angular velocity: omega_body = R^T @ omega_world.
            # Start each repeat at a uniformly random orientation so the
            # observer samples different parts of the panorama across repeats.
            R = _random_rotation_matrix(rng)
            rotated_xyz = np.zeros((time_per_stim, num_cells, 3))
            omega_body = np.zeros((time_per_stim, 3))

            for ti in range(time_per_stim):
                omega_world = omega_seg[:, ti]
                R = _rotation_increment(omega_world, dt) @ R
                rotated_xyz[ti] = (R @ sample_xyz.T).T
                omega_body[ti] = R.T @ omega_world

            # --- sample the filtered panoramic image (out-of-bounds -> mean) ---
            stim, out_of_bounds = _sample_pano(pano_filtered, rotated_xyz, pano_y_lim, pano_x_lim)
            stimulus_list.append(stim)
            omega_body_list.append(omega_body)
            oob_list.append(out_of_bounds)

            # --- compute optic flow (batched over time) ---
            omega_body_t = torch.tensor(omega_body, dtype=torch.float32)
            T_zeros = T_zero.expand(omega_body_t.shape[0], -1)
            u_right, u_up = generate_optic_flow(phi_t, theta_t, omega_body_t, T_zeros, 1.0)
            # generate_optic_flow returns rad/s; convert to deg/s
            of = np.stack([u_right.numpy(), u_up.numpy()], axis=-1) * (180.0 / np.pi)
            # Mark optic flow as NaN where stimulus is out-of-bounds (constant),
            # since the model cannot extract motion from a constant stimulus.
            # NaN-aware loss functions will ignore these positions.
            of[out_of_bounds, :] = np.nan
            optic_flow_list.append(of)

    # --- stack into final arrays ---
    stimulus = np.stack(stimulus_list)             # (num_panos * num_repeats_total, time_per_stim, num_cells)
    optic_flow = np.stack(optic_flow_list)         # (num_panos * num_repeats_total, time_per_stim, num_cells, 2)
    rotation_vectors = np.stack(omega_body_list)   # (num_panos * num_repeats_total, time_per_stim, 3)
    oob_mask = np.stack(oob_list)                  # (num_panos * num_repeats_total, time_per_stim, num_cells)

    return stimulus, optic_flow, rotation_vectors, oob_mask


def _omega_body_cardinal_pano(direction, velocity_deg_per_s):
    """Body-frame angular velocity (rad/s) for constant panoramic rotation.

    Coordinate convention matches ``get_random_velocity_pano`` and
    ``make_lr_sine_wave`` / ``generate_optic_flow``: +x forward, +y left, +z up.
    Signs are chosen so horizontal motion matches LR sine (yaw, about body z)
    and vertical motion matches the retinal up component from pitch (about
    body y).
    """
    w = float(velocity_deg_per_s) * (np.pi / 180.0)
    d = direction.lower()
    if d == "right":
        return np.array([0.0, 0.0, w], dtype=np.float64)
    if d == "left":
        return np.array([0.0, 0.0, -w], dtype=np.float64)
    if d == "up":
        return np.array([0.0, w, 0.0], dtype=np.float64)
    if d == "down":
        return np.array([0.0, -w, 0.0], dtype=np.float64)
    raise ValueError(f"Unknown direction {direction!r}; expected left/right/up/down")


def get_constant_velocity_pano(
    sample_phi,
    sample_theta,
    direction,
    velocity_deg_per_s,
    dt,
    stim_time,
    pre_stim_time=0.0,
    seed=None,
    pano_indices=None,
    log_scale=False,
    data_folder=None,
):
    """
    Panoramic natural scenes rotating at constant angular speed about a
    cardinal body axis (yaw for left/right, pitch for up/down).

    Uses the same loading, Gaussian filtering, and spherical sampling as
    ``get_random_velocity_pano``, but with a fixed body-frame angular
    velocity during the stimulus interval, so the fly always turns about its
    own z (yaw) or y (pitch) axis whatever its random initial orientation.
    Pre-stimulus time holds the view fixed at that random initial orientation
    (reproducible from ``seed``).

    Args:
        sample_phi: (N,) azimuthal angles (radians).
        sample_theta: (N,) polar angles (radians).
        direction: one of ``"left"``, ``"right"``, ``"up"``, ``"down"``.
        velocity_deg_per_s: rotation speed (deg/s); sign is taken from
            ``direction``.
        dt: time step (seconds).
        stim_time: motion duration (seconds), excluding ``pre_stim_time``.
        pre_stim_time: static panorama duration before rotation (seconds).
        seed: RNG seed for initial orientations (one per panorama).
        pano_indices: optional 1D array of indices into ``all_panoramic.pt``;
            default uses every panorama.
        log_scale: if True, log-compress the raw panoramas
            (``log(x + 1) / log(max + 1)``) before filtering.
        data_folder: data root holding ``stimulus/baked/all_panoramic.pt``;
            defaults to ``[data].data_folder`` of the working-directory
            ``config.toml``.

    Returns:
        stimulus: (P, T_total, N) with ``T_total = pre + stim`` steps, values
            in ``[0, 1]`` as in ``get_random_velocity_pano``.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    omega_body = _omega_body_cardinal_pano(direction, velocity_deg_per_s)
    R_inc = _rotation_increment(omega_body, dt)

    raw_panos = _load_all_panos(data_folder=data_folder)
    if log_scale:
        raw_panos = np.log(raw_panos+1) / np.log(raw_panos.max()+1)
    if pano_indices is not None:
        raw_panos = raw_panos[np.asarray(pano_indices, dtype=int)]

    sample_xyz = np.stack([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta),
    ], axis=1).astype(np.float64)

    pre_steps = int(round(pre_stim_time / dt))
    stim_steps = int(round(stim_time / dt))

    pano_list = []
    for i in range(raw_panos.shape[0]):
        pano_filtered, pano_y_lim, pano_x_lim = _filter_pano(raw_panos[i])

        R = _random_rotation_matrix(rng)

        rows = []
        pre_row, _ = _sample_pano(pano_filtered, (R @ sample_xyz.T).T, pano_y_lim, pano_x_lim)
        for _ in range(pre_steps):
            rows.append(pre_row.copy())

        for _ti in range(stim_steps):
            # Right-multiplying applies the increment in the body frame, so the
            # rotation axis stays the fly's own yaw/pitch axis.
            R = R @ R_inc
            xyz = (R @ sample_xyz.T).T
            rows.append(_sample_pano(pano_filtered, xyz, pano_y_lim, pano_x_lim)[0])

        pano_list.append(np.stack(rows, axis=0))

    return np.stack(pano_list, axis=0)


def consolidate_stim(stim_dict, optic_flow_dict):
    time_per_stim, num_cell = next(iter(stim_dict.values()))[0].shape[1:]
    stimulus = np.zeros((0, time_per_stim, num_cell))
    optic_flow = np.zeros((0, time_per_stim, num_cell, 2))

    for b in stim_dict:
        this_stim = np.concatenate(stim_dict[b])
        stimulus = np.concatenate((stimulus, this_stim), axis=0)
        this_optic_flow = np.concatenate(optic_flow_dict[b])
        optic_flow = np.concatenate((optic_flow, this_optic_flow))

    return stimulus, optic_flow


def _consolidate_ego(rotation_dict, translation_dict):
    """Stack per-block ego-motion lists into ``(examples, time, 3)`` arrays.

    Mirrors :func:`consolidate_stim`'s block ordering so the ego-motion examples
    line up one-to-one with the consolidated stimulus/optic-flow examples.
    """
    rotation = np.concatenate(
        [np.concatenate(rotation_dict[b]) for b in rotation_dict], axis=0)
    translation = np.concatenate(
        [np.concatenate(translation_dict[b]) for b in translation_dict], axis=0)
    return rotation, translation

@dataclass
class StimulusData:
    """Unified return type for :func:`get_stimulus`.

    ``stimulus`` is ``(examples, time, num_eyes * N)``; for two eyes the columns
    are the right eye (sampled at ``(phi, theta)``) followed by the left eye
    (``(-phi, theta)``). ``optic_flow`` is the same with a trailing ``(2,)``.
    ``rotation`` / ``translation`` are ``(examples, time, 3)`` observer ego-motion
    (deg/s and velocity), or ``None`` for one-eye stimuli, which carry no
    ego-motion target. ``eye_offset`` is the right-eye camera offset of the rig
    that produced the flow (``None`` for one eye).

    ``group_id`` is a ``(examples,)`` integer tensor identifying the source each
    example was derived from (e.g. the raw fly-view clip a segment was cut
    from, all repeats of one panorama, or the phase/contrast variants of one
    grating condition). Examples that share a ``group_id`` are correlated and
    must stay together when splitting train/test, so the split happens at
    whole-group granularity. Generators that produce independent examples give
    each example its own ``group_id``.
    """
    stimulus: torch.Tensor
    optic_flow: torch.Tensor
    train_mask: torch.Tensor
    rotation: torch.Tensor | None
    translation: torch.Tensor | None
    num_eyes: int
    eye_offset: tuple[float, float, float] | None
    group_id: torch.Tensor


@dataclass
class _GenContext:
    """Everything a per-stimulus generator needs, plus eye-layout helpers."""
    config: dict
    data_config: dict
    column_phi_theta: object
    num_eyes: int
    model_dt: float
    stim_time: float
    seed: object
    device: torch.device
    dtype: torch.dtype
    sine_direction: str | None = None

    @property
    def phi(self):
        """Azimuthal sample angles for the requested eyes (right, then -right)."""
        phi = self.column_phi_theta[:, 0]
        return np.concatenate([phi, -phi]) if self.num_eyes == 2 else phi

    @property
    def theta(self):
        """Polar sample angles for the requested eyes (shared across eyes)."""
        theta = self.column_phi_theta[:, 1]
        return np.concatenate([theta, theta]) if self.num_eyes == 2 else theta

    @property
    def n_cols(self):
        return self.num_eyes * self.column_phi_theta.shape[0]

    def require_columns(self, name):
        if self.column_phi_theta is None:
            raise ValueError(f"{name} requires column_phi_theta input")

    def tensor(self, arr):
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype)


def _ones_mask(stim):
    return torch.ones(stim.shape[:3], dtype=torch.bool, device=stim.device)


def _zero_ego(stim):
    """Zero rotation/translation (examples, time, 3) matching ``stim``."""
    z = torch.zeros((stim.shape[0], stim.shape[1], 3),
                    device=stim.device, dtype=stim.dtype)
    return z, z.clone()


# --- Per-stimulus generators -------------------------------------------------
# Each takes a _GenContext and returns
# ``(stimulus, optic_flow, rotation, translation, train_mask)`` tensors for the
# requested eye count, optionally followed by a ``(examples,)`` ``group_id``
# that keeps correlated examples on one side of the train/test split (see
# StimulusData). Generators with no ego-motion return zeros for
# rotation/translation (nulled to ``None`` for one-eye output by get_stimulus).

def _gen_static_sine(ctx):
    ctx.require_columns('static_sine')
    stim, flow, rot, trans, group_id = create_balanced_sine_wave(
        ctx.phi, ctx.theta,
        temporal_frequency=ctx.data_config['static_sine']['temporal_frequencies'],
        dt=ctx.model_dt, stim_time=ctx.stim_time, pre_stim_time=0.0, post_stim_time=0.0,
    )
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot, trans = ctx.tensor(rot), ctx.tensor(trans)
    return stim, flow, rot, trans, _ones_mask(stim), torch.as_tensor(group_id)


def _gen_directional_sine(ctx):
    ctx.require_columns('directional_sine')
    if ctx.sine_direction is None:
        raise ValueError('directional_sine requires sine_direction input')
    stim, flow, rot, trans = create_directional_sine_wave(
        ctx.phi, ctx.theta, direction=ctx.sine_direction,
        dt=ctx.model_dt, stim_time=ctx.stim_time, pre_stim_time=0.0, post_stim_time=0.0,
    )
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot, trans = ctx.tensor(rot), ctx.tensor(trans)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_random_sine(ctx):
    ctx.require_columns('random_sine')
    p = ctx.data_config['random_sine']
    stim, flow, rot, trans, group_id = create_random_tf_sine_wave(
        ctx.phi, ctx.theta, velocity_amp=p['velocity_amp'],
        velocity_time_scale=p['velocity_time_scale'], num_repeats=p['num_repeats'],
        dt=ctx.model_dt, stim_time=ctx.stim_time, seed=ctx.seed,
    )
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot, trans = ctx.tensor(rot), ctx.tensor(trans)
    return stim, flow, rot, trans, _ones_mask(stim), torch.as_tensor(group_id)


def _gen_edge(ctx):
    ctx.require_columns('edge')
    # make_lr_edge / make_ud_edge are built for one eye's columns: over both
    # eyes the roll-based UD sweep starts already past the left-eye columns,
    # and slow LR edges never reach one eye within stim_time. So generate the
    # right eye and give the left eye its sagittal mirror image (same value at
    # (-phi, theta)): LR edges then move front-to-back (or back-to-front) in
    # both eyes and UD edges move the same way vertically in both.
    stim, flow = create_edge_stimulus(
        ctx.column_phi_theta[:, 0], ctx.column_phi_theta[:, 1],
        velocities=ctx.data_config['edge']['edge_velocities'],
        dt=ctx.model_dt, stim_time=ctx.stim_time, pre_stim_time=0.0, post_stim_time=0.0,
    )
    if ctx.num_eyes == 2:
        stim = np.concatenate([stim, stim], axis=2)
        flow = np.concatenate([flow, flow], axis=2)  # placeholder zeros
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot, trans = _zero_ego(stim)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_random_pano(ctx):
    ctx.require_columns('random_pano')
    p = ctx.data_config['random_pano']
    stim, flow, rotation_vectors, _ = get_random_velocity_pano(
        ctx.phi, ctx.theta, velocity_amp=p['velocity_amp'],
        velocity_time_scale=p['velocity_time_scale'], num_repeats=p['num_repeats'],
        dt=ctx.model_dt, stim_time=ctx.stim_time, seed=ctx.seed,
        data_folder=get_data_folder(ctx.config),
    )
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot = ctx.tensor(rotation_vectors * 180.0 / np.pi)  # rad/s -> deg/s
    trans = torch.zeros_like(rot)
    # Examples are panorama-major with 2 * num_repeats (symmetrised) repeats
    # per panorama; all repeats of one panorama share a group.
    group_id = torch.arange(stim.shape[0]) // (2 * p['num_repeats'])
    return stim, flow, rot, trans, _ones_mask(stim), group_id


def _gen_counterphase_sine(ctx):
    ctx.require_columns('counterphase_sine')
    stim, flow = create_counterphase_sine(
        ctx.phi, ctx.theta,
        temporal_frequencies=ctx.data_config['counterphase_sine']['temporal_frequencies'],
        seed=ctx.seed, dt=ctx.model_dt, stim_time=ctx.stim_time,
        pre_stim_time=0.0, post_stim_time=0.0,
    )
    stim, flow = ctx.tensor(stim), ctx.tensor(flow)
    rot, trans = _zero_ego(stim)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_full_field_step(ctx):
    ctx.require_columns('full_field_step')
    n_time = int(round(ctx.stim_time / ctx.model_dt))
    n_cols = ctx.n_cols
    stim = torch.zeros((3, n_time, n_cols), device=ctx.device, dtype=ctx.dtype)
    stim[0] = 0.0
    stim[1] = 0.5
    stim[2] = 1.0
    flow = torch.zeros((3, n_time, n_cols, 2), device=ctx.device, dtype=ctx.dtype)
    rot, trans = _zero_ego(stim)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_full_field_flicker(ctx):
    ctx.require_columns('full_field_flicker')
    n_cols = ctx.n_cols
    flicker_tfs = ctx.data_config['full_field_flicker']['flicker_temporal_frequencies']
    if not isinstance(flicker_tfs, list):
        flicker_tfs = [flicker_tfs]
    stim_arrays, flow_arrays = [], []
    for tf in flicker_tfs:
        s, f = make_full_field_flicker(n_cols, tf, ctx.model_dt, ctx.stim_time)
        stim_arrays.append(s[None])
        flow_arrays.append(f[None])
    stim = ctx.tensor(np.concatenate(stim_arrays))
    flow = ctx.tensor(np.concatenate(flow_arrays))
    rot, trans = _zero_ego(stim)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_flyview(ctx):
    ctx.require_columns('flyview')
    two = ctx.num_eyes == 2
    data_path = get_data_folder(ctx.config) / 'stimulus' / 'baked' / 'flyview.pt'
    if two:
        stim_np, flow_np, rot_np, trans_np, clip_ids = load_fly_view_dataset(
            ctx.model_dt, ctx.stim_time, two_eyes=True, data_path=data_path)
    else:
        stim_np, flow_np, clip_ids = load_fly_view_dataset(
            ctx.model_dt, ctx.stim_time, two_eyes=False, data_path=data_path)
    if stim_np.shape[2] != ctx.n_cols:
        raise ValueError(
            f"flyview columns ({stim_np.shape[2]}) != num_eyes*N ({ctx.n_cols})")
    stim, flow = ctx.tensor(stim_np), ctx.tensor(flow_np)
    if two:
        rot, trans = ctx.tensor(rot_np), ctx.tensor(trans_np)
    else:
        rot, trans = _zero_ego(stim)
    # Segments cut from the same raw clip share a group id so they cannot be
    # split across train/test (avoids leaking correlated frames).
    group_id = torch.as_tensor(clip_ids, device=stim.device, dtype=torch.long)
    return stim, flow, rot, trans, _ones_mask(stim), group_id


def _gen_expanding_dot(ctx):
    ctx.require_columns('expanding_dot')
    p = ctx.data_config['expanding_dot']
    stim_np, flow_np = create_expanding_dot_stimulus(
        ctx.phi, ctx.theta, dt=ctx.model_dt, stim_time=ctx.stim_time,
        pre_stim_time=0.0, post_stim_time=0.0, seed=ctx.seed,
        num_light=int(p['num_light']), num_dark=int(p['num_dark']),
        filter_std=float(p['filter_std']),
    )
    stim, flow = ctx.tensor(stim_np), ctx.tensor(flow_np)
    rot, trans = _zero_ego(stim)
    return stim, flow, rot, trans, _ones_mask(stim)


def _gen_small_drone(ctx):
    ctx.require_columns('small_drone')
    return small_drone(ctx.config, ctx.column_phi_theta, ctx.model_dt,
                       ctx.device, ctx.dtype, ctx.stim_time)


def _gen_rpi_drone(ctx):
    ctx.require_columns('rpi_drone')
    return rpi_drone(ctx.config, ctx.column_phi_theta, ctx.model_dt,
                     ctx.device, ctx.dtype, ctx.stim_time)


# Registry: stimulus_type -> (generator, supported eye counts). The first
# supported eye count is the type's natural default (see ``default_num_eyes``).
# Every type defaults to two eyes, so create_training_data always bakes
# two-eye data; training then runs the two-eye forward path for it.
_STIMULUS_REGISTRY: dict[str, tuple] = {
    'static_sine': (_gen_static_sine, (2, 1)),
    'directional_sine': (_gen_directional_sine, (2, 1)),
    'random_sine': (_gen_random_sine, (2, 1)),
    'edge': (_gen_edge, (2, 1)),
    'random_pano': (_gen_random_pano, (2, 1)),
    'counterphase_sine': (_gen_counterphase_sine, (2, 1)),
    'full_field_step': (_gen_full_field_step, (2, 1)),
    'full_field_flicker': (_gen_full_field_flicker, (2, 1)),
    'flyview': (_gen_flyview, (2, 1)),
    'expanding_dot': (_gen_expanding_dot, (2, 1)),
    'small_drone': (_gen_small_drone, (2,)),
    'rpi_drone': (_gen_rpi_drone, (2,)),
}

# Registered types that only analysis scripts may generate; create_training_data
# refuses to bake them, giving the reason below.
ANALYSIS_ONLY_STIMULUS_TYPES: dict[str, str] = {
    'edge': "an analysis-only probe whose optic-flow targets are placeholder zeros",
    'expanding_dot': "an analysis-only probe whose optic-flow targets are placeholder zeros",
    'directional_sine': "generated one direction at a time (it needs a sine_direction, "
                        "which only the analysis scripts pass); bake static_sine instead",
}


def bakeable_stimulus_types() -> list[str]:
    """Stimulus types ``create_training_data`` can bake into training data."""
    return sorted(t for t in _STIMULUS_REGISTRY if t not in ANALYSIS_ONLY_STIMULUS_TYPES)


def default_num_eyes(stimulus_type: str) -> int:
    """Natural eye count for ``stimulus_type`` (first the generator supports)."""
    if stimulus_type not in _STIMULUS_REGISTRY:
        raise ValueError(
            f"Unknown stimulus_type '{stimulus_type}'. "
            f"Expected one of: {', '.join(sorted(_STIMULUS_REGISTRY))}.")
    return _STIMULUS_REGISTRY[stimulus_type][1][0]


def get_stimulus(
    stimulus_type: str | list[str],
    config,
    column_phi_theta=None,
    *,
    num_eyes: int | None = None,
    data_config=None,
    sine_direction: str | None = None,
) -> StimulusData:
    """Generate a stimulus (or a list of types, concatenated) in a unified format.

    Single entry point replacing the old ``get_stimulus_one_eye`` /
    ``get_stimulus_two_eyes`` split. It dispatches to the per-type generator in
    ``_STIMULUS_REGISTRY`` for the requested ``num_eyes``:

    * ``num_eyes=1`` -> right eye only, ``N`` columns sampled at ``(phi, theta)``.
    * ``num_eyes=2`` -> right then left, ``2*N`` columns (left at ``(-phi, theta)``),
      with observer ego-motion targets.

    When ``num_eyes`` is None it defaults to each type's natural eye count
    (:func:`default_num_eyes`). Stimulus-generation parameters come from
    ``data_config`` (defaults to ``config['data']``); model ``dt`` / ``device``
    / ``dtype`` and the data folder (for the file-backed types) come from
    ``config``. ``sine_direction`` is required by ``directional_sine``.

    Returns a :class:`StimulusData`. For one-eye output ``rotation`` /
    ``translation`` / ``eye_offset`` are ``None``.
    """
    if data_config is None:
        data_config = config['data']
    shared = data_config['shared']

    stimulus_types = [stimulus_type] if isinstance(stimulus_type, str) else list(stimulus_type)
    if not stimulus_types:
        raise ValueError("stimulus_type must be a non-empty type or list of types.")
    for st in stimulus_types:
        if st not in _STIMULUS_REGISTRY:
            raise ValueError(
                f"Unknown stimulus_type '{st}'. "
                f"Expected one of: {', '.join(sorted(_STIMULUS_REGISTRY))}.")

    if num_eyes is None:
        num_eyes = default_num_eyes(stimulus_types[0])
    for st in stimulus_types:
        supported = _STIMULUS_REGISTRY[st][1]
        if num_eyes not in supported:
            raise ValueError(
                f"Stimulus '{st}' does not support num_eyes={num_eyes} "
                f"(supports {supported}).")

    device = torch.device(config['model']['device'])
    dtype = getattr(torch, config['model']['dtype'])
    model_dt = config['model']['dt']
    pre_stim_time = shared['pre_stim_time']

    ctx = _GenContext(
        config=config, data_config=data_config, column_phi_theta=column_phi_theta,
        num_eyes=num_eyes, model_dt=model_dt, stim_time=shared['stim_time'],
        seed=shared['seed'], device=device, dtype=dtype,
        sine_direction=sine_direction,
    )

    stim_list, flow_list, rot_list, trans_list, mask_list = [], [], [], [], []
    group_list = []
    group_offset = 0
    for st in stimulus_types:
        generator = _STIMULUS_REGISTRY[st][0]
        result = generator(ctx)
        # Generators may optionally return a per-example group id (6th element);
        # those that don't get each example as its own group.
        if len(result) == 6:
            stim, flow, rot, trans, mask, group_id = result
            group_id = group_id.to(device=stim.device, dtype=torch.long)
        else:
            stim, flow, rot, trans, mask = result
            group_id = torch.arange(stim.shape[0], device=stim.device, dtype=torch.long)
        # Offset so group ids stay unique across concatenated stimulus types.
        group_list.append(group_id + group_offset)
        if group_id.numel():
            group_offset += int(group_id.max()) + 1
        stim_list.append(stim)
        flow_list.append(flow)
        rot_list.append(rot)
        trans_list.append(trans)
        mask_list.append(mask)

    stimulus = torch.cat(stim_list, dim=0)
    optic_flow = torch.cat(flow_list, dim=0)
    rotation = torch.cat(rot_list, dim=0)
    translation = torch.cat(trans_list, dim=0)
    train_mask = torch.cat(mask_list, dim=0)
    group_id = torch.cat(group_list, dim=0)

    # Prepend the stationary pre-stimulus interval (mean-gray, zero flow/ego).
    pre_stim_points = int(round(pre_stim_time / model_dt))
    if pre_stim_points > 0:
        n = stimulus.shape[0]
        num_cols = stimulus.shape[2]
        pre_stim = torch.full((n, pre_stim_points, num_cols), 0.5, device=device, dtype=dtype)
        pre_flow = torch.zeros((n, pre_stim_points, optic_flow.shape[2], optic_flow.shape[3]),
                               device=device, dtype=dtype)
        pre_rot = torch.zeros((n, pre_stim_points, 3), device=device, dtype=dtype)
        pre_mask = torch.zeros((n, pre_stim_points, num_cols), dtype=torch.bool, device=device)
        stimulus = torch.cat([pre_stim, stimulus], dim=1)
        optic_flow = torch.cat([pre_flow, optic_flow], dim=1)
        rotation = torch.cat([pre_rot, rotation], dim=1)
        translation = torch.cat([pre_rot.clone(), translation], dim=1)
        train_mask = torch.cat([pre_mask, train_mask], dim=1)

    if num_eyes == 1:
        return StimulusData(
            stimulus=stimulus, optic_flow=optic_flow, train_mask=train_mask,
            rotation=None, translation=None, num_eyes=1, eye_offset=None,
            group_id=group_id,
        )
    return StimulusData(
        stimulus=stimulus, optic_flow=optic_flow, train_mask=train_mask,
        rotation=rotation, translation=translation, num_eyes=2,
        eye_offset=camera_offset_for_stimulus(stimulus_types),
        group_id=group_id,
    )


# Right-eye camera offset (body frame +x forward / +y left / +z up) per two-eye
# stimulus type. Its length unit must match the dataset's translation targets
# (metres for the drone rigs, whose translation is in m/s; depth units for
# flyview), because the two-eye solver's absolute translation scales with it.
# The offset is a property of the rig that produced the flow: a single-viewpoint
# render has none, a real two-camera rig carries its physical half-baseline.
# Populated by the @camera_offset decorator on each stimulus loader (keyed by
# loader name = stimulus_type); loaders without the decorator are assumed
# (0, 0, 0). The left-eye offset is the sagittal mirror, applied by
# OpticLobeRegion.mirror_camera_offset.
STIMULUS_CAMERA_OFFSETS: dict[str, tuple[float, float, float]] = {}


def camera_offset(x=0.0, y=0.0, z=0.0):
    """Decorator: register a stimulus loader's right-eye camera offset.

    The loader's ``__name__`` is used as the ``stimulus_type`` key. A loader
    without this decorator defaults to ``(0, 0, 0)`` (single viewpoint, no
    parallax baseline).
    """
    def decorator(func):
        STIMULUS_CAMERA_OFFSETS[func.__name__] = (float(x), float(y), float(z))
        return func
    return decorator


def camera_offset_for_stimulus(stimulus_type: str | list[str]):
    """Right-eye camera offset for a two-eye ``stimulus_type`` (or list).

    Returns a length-3 tuple, defaulting to ``(0, 0, 0)`` for any type whose
    loader was not decorated with ``@camera_offset``. A list of types must share
    one offset (you cannot train a single rig geometry on a mix of rigs);
    otherwise raises.
    """
    types = [stimulus_type] if isinstance(stimulus_type, str) else list(stimulus_type)
    offsets = {STIMULUS_CAMERA_OFFSETS.get(t, (0.0, 0.0, 0.0)) for t in types}
    if len(offsets) > 1:
        raise ValueError(
            f"Mixed camera offsets across stimulus types {types}: {sorted(offsets)}. "
            "Train one rig geometry at a time."
        )
    return offsets.pop()


def small_drone(config, column_phi_theta, model_dt, device, dtype, stim_time):
    """Two-eye loader for the equirectangular ``small_drone`` rig.

    Loads the cached clips from ``<data_folder>/stimulus/baked/small_drone.pt``
    (built by ``process_raw_data.py``) via
    :func:`drone_utilities.get_drone_stimulus`.

    Returns ``(stimulus, optic_flow, observer_rotation, observer_translation,
    train_mask, group_id)`` (see :func:`drone_utilities.drone_clips_to_two_eye`).
    """
    drone_clips = du.get_drone_stimulus(
        model_dt, dataset='small_drone',
        data_folder=get_data_folder(config),
        sample_phi=column_phi_theta[:, 0], sample_theta=column_phi_theta[:, 1],
    )
    return du.drone_clips_to_two_eye(drone_clips, stim_time, model_dt, device, dtype)


@camera_offset(0.050, -0.050, 0.013)
def rpi_drone(config, column_phi_theta, model_dt, device, dtype, stim_time):
    """Two-eye loader for the ``rpi_drone`` dataset.

    A real two-camera rig; the ``@camera_offset`` above declares its right-eye
    half-baseline in metres, matching the m/s ground-truth translation. Loads
    the cached clips from ``<data_folder>/stimulus/baked/rpi_drone.pt`` (built
    by ``process_raw_data.py``) via :func:`drone_utilities.get_drone_stimulus`.

    Returns ``(stimulus, optic_flow, observer_rotation, observer_translation,
    train_mask, group_id)`` (see :func:`drone_utilities.drone_clips_to_two_eye`).
    """
    drone_clips = du.get_drone_stimulus(
        model_dt, dataset='rpi_drone',
        data_folder=get_data_folder(config),
        sample_phi=column_phi_theta[:, 0], sample_theta=column_phi_theta[:, 1],
    )
    return du.drone_clips_to_two_eye(drone_clips, stim_time, model_dt, device, dtype)
