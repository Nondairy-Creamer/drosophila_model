"""Build a relative top-down depth map of the room from drone flow.

Loads a trained model, runs it on a drone clip to obtain rotational and
translational ego motion plus per-column nearness rho ~ ||T||/Z, refines
the per-frame motion via photometric matching of consecutive frames on
the ommatidium grid, integrates ego motion to a 3D pose track, and
accumulates a top-down 2D map of where the room's surfaces are.

Three pose tracks are computed for diagnostic comparison:

    Pose A: integrated from the model's omega and T directly.
    Pose B: integrated from photometrically-refined omega and t_hat
            (with the model's per-frame ||T|| supplying scale).
    Pose C: integrated from the per-frame average of A's and B's rates.

The depth map is built using Pose B (or Pose A if photometric is disabled).
Monocular flow is fundamentally up-to-scale, so the map is "relative":
distances are in the same arbitrary units as the model's flow-derived T.

Usage:
    python analysis/make_map.py models/my_model/
    python analysis/make_map.py models/my_model/ --clip-index 1 --max-frames 300
"""
import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _repo_root)

import numpy as np
import torch
from scipy.spatial import cKDTree

import data_utilities as du
import neural_models as nm
import stimulus_utilities as su


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a relative top-down depth map from drone flow.",
    )
    parser.add_argument("model_folder", type=str,
                        help="Path to model folder or model.pt")
    parser.add_argument("--clip-index", type=int, default=0,
                        help="Drone clip index to process (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model_folder>/depth_map)")
    parser.add_argument("--no-photometric", action="store_true",
                        help="Skip photometric refinement; use the model's "
                             "ego motion directly.")
    parser.add_argument("--photo-iters", type=int, default=80,
                        help="Number of optimizer iterations for photometric refinement.")
    parser.add_argument("--photo-lr", type=float, default=1e-2,
                        help="Adam learning rate for photometric refinement.")
    parser.add_argument("--photo-smooth", type=float, default=0.0,
                        help="Temporal smoothness weight on photometric rates.")
    parser.add_argument("--grid-size", type=int, default=200,
                        help="Top-down grid resolution (cells per side).")
    parser.add_argument("--depth-percentile", type=float, default=98.0,
                        help="Percentile of column distances used to bound the map.")
    parser.add_argument("--n-azimuth", type=int, default=360,
                        help="Number of azimuth bins per frame (1 per degree).")
    parser.add_argument("--azimuth-sigma-deg", type=float, default=5.0,
                        help="Gaussian RBF bandwidth (deg) for azimuth resampling.")
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    model_folder = model_path.parent if model_path.is_file() else model_path
    local_config = du.load_config()

    print(f"Loading model from {model_path}...")
    model = nm.NeuralModel.load(model_path)
    model.eval()
    model_dt = model.dt

    connectome_folder = Path(local_config["data"]["connectome_folder"])
    drone_data_path = connectome_folder / "drone_data.pt"

    print("Loading drone clips...")
    drone_clips = su.get_drone_stimulus(model_dt, data_path=str(drone_data_path))
    if args.clip_index >= len(drone_clips):
        raise ValueError(
            f"Clip {args.clip_index} requested but only {len(drone_clips)} "
            f"clips available."
        )

    clip = drone_clips[args.clip_index]
    clip_id = clip["clip_id"]
    N = clip["right"].shape[1]
    num_frames = clip["right"].shape[0]
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    output_dir = (Path(args.output_dir) if args.output_dir
                  else model_folder / "depth_map")
    output_dir.mkdir(parents=True, exist_ok=True)

    right_input = torch.tensor(
        clip["right"][:num_frames], device=model.device, dtype=model.dtype,
    ).unsqueeze(0)
    left_input = torch.tensor(
        clip["left"][:num_frames], device=model.device, dtype=model.dtype,
    ).unsqueeze(0)

    in_fov_right = clip["in_fov_right"].numpy()
    in_fov_left = clip["in_fov_left"].numpy()
    in_fov_2e = np.concatenate([in_fov_right, in_fov_left])
    oob = torch.tensor(~in_fov_2e, device=model.device)

    print(f"Clip {args.clip_index} (id={clip_id}): "
          f"running model on {num_frames} frames, N={N}...")
    with torch.no_grad():
        pred_rot, pred_trans, right_res, left_res = (
            model.estimate_ego_motion_from_image(
                right_input, left_input, oob_columns=oob,
            )
        )

    rho_full, valid_mask = _per_column_rho(
        model, right_res, left_res, oob, num_frames,
    )
    omega_model = pred_rot[0].cpu().numpy().astype(np.float64)   # (T, 3) deg/s
    T_model = pred_trans[0].cpu().numpy().astype(np.float64)      # (T, 3) flow-units
    rho = rho_full.cpu().numpy().astype(np.float64)               # (T, 2N)
    valid_np = valid_mask.cpu().numpy()                            # (2N,)

    stim = np.concatenate(
        [clip["right"][:num_frames], clip["left"][:num_frames]],
        axis=1,
    ).astype(np.float64)                                            # (T, 2N)

    phi_2e = model.column_phi_theta_two_eyes[0].cpu().numpy()
    theta_2e = model.column_phi_theta_two_eyes[1].cpu().numpy()
    x_2e = np.stack([
        np.sin(theta_2e) * np.cos(phi_2e),
        np.sin(theta_2e) * np.sin(phi_2e),
        np.cos(theta_2e),
    ], axis=-1).astype(np.float64)                                  # (2N, 3)

    # ------------------------------------------------------------------
    # Photometric refinement
    # ------------------------------------------------------------------
    if not args.no_photometric:
        print("Photometric pose refinement on the ommatidium grid...")
        omega_phot, T_phot, photo_residual_pre, photo_residual_post = photometric_refine(
            stim=stim, x_2e=x_2e, rho=rho, valid=valid_np,
            omega_init=omega_model, T_init=T_model, dt=model_dt,
            num_iters=args.photo_iters, lr=args.photo_lr,
            smooth=args.photo_smooth, device=model.device,
        )
    else:
        print("Skipping photometric refinement.")
        omega_phot = omega_model.copy()
        T_phot = T_model.copy()
        photo_residual_pre = np.zeros(num_frames - 1)
        photo_residual_post = np.zeros(num_frames - 1)

    # ------------------------------------------------------------------
    # Three pose tracks
    # ------------------------------------------------------------------
    pose_A = _integrate_pose(omega_model, T_model, model_dt)
    pose_B = _integrate_pose(omega_phot, T_phot, model_dt)
    omega_fused = 0.5 * (omega_model + omega_phot)
    T_fused = 0.5 * (T_model + T_phot)
    pose_C = _integrate_pose(omega_fused, T_fused, model_dt)

    # ------------------------------------------------------------------
    # Top-down depth map (uses Pose B, which is the photometrically-
    # refined track when available; falls back to Pose A when not).
    # ------------------------------------------------------------------
    print("Building top-down room map from per-frame azimuth resampling...")
    grid = build_room_map(
        rho=rho, x_2e=x_2e, valid=valid_np,
        omega=omega_phot, T=T_phot, dt=model_dt,
        n_azimuth=args.n_azimuth, sigma_deg=args.azimuth_sigma_deg,
        grid_size=args.grid_size, depth_percentile=args.depth_percentile,
    )

    # ------------------------------------------------------------------
    # Save arrays + plots
    # ------------------------------------------------------------------
    out_npz = output_dir / f"make_map_clip_{args.clip_index}.npz"
    np.savez(
        out_npz,
        clip_id=np.int64(clip_id),
        model_dt=np.float64(model_dt),
        omega_model=omega_model, T_model=T_model,
        omega_phot=omega_phot, T_phot=T_phot,
        omega_fused=omega_fused, T_fused=T_fused,
        rho=rho, valid_mask=valid_np,
        photo_residual_pre=photo_residual_pre,
        photo_residual_post=photo_residual_post,
        pose_A_pos=pose_A["pos"], pose_A_rot=pose_A["rot"],
        pose_B_pos=pose_B["pos"], pose_B_rot=pose_B["rot"],
        pose_C_pos=pose_C["pos"], pose_C_rot=pose_C["rot"],
        grid_x_edges=grid["grid_x_edges"],
        grid_y_edges=grid["grid_y_edges"],
        grid_weight=grid["grid_weight"],
        grid_nearness=grid["grid_nearness"],
        scatter_xy=grid["scatter_xy"],
        scatter_rho=grid["scatter_rho"],
        scatter_Z=grid["scatter_Z"],
        scatter_weight=grid["scatter_weight"],
        scatter_frame=grid["scatter_frame"],
        scatter_azimuth=grid["scatter_azimuth"],
        trajectory_xy=grid["trajectory_xy"],
    )
    print(f"  Saved arrays -> {out_npz}")

    plot_topdown_map(
        grid, pose_A, pose_B, pose_C,
        output_dir / f"make_map_clip_{args.clip_index}.png",
        clip_idx=args.clip_index, clip_id=clip_id,
    )
    plot_diagnostics(
        omega_model, omega_phot,
        T_model, T_phot,
        photo_residual_pre, photo_residual_post,
        model_dt,
        output_dir / f"make_map_clip_{args.clip_index}_diagnostics.png",
        clip_idx=args.clip_index, clip_id=clip_id,
    )


# ============================================================================
# Helpers shared with movie_scripts/run_model_on_drone.py (kept local to
# avoid importing from a sibling script).
# ============================================================================

def _per_column_rho(model, right_res, left_res, oob, num_frames):
    """Run the ego-motion solver to recover per-column rho_i (nearness) for
    both eyes. Invalid (OOB or NaN-flow) columns have NaN rho.

    Returns:
        rho_full: (T, 2N) tensor of per-point ||T||/Z.
        valid_mask: (2N,) bool tensor of always-valid columns.
    """
    decoder_key = next(
        k for k, m in model._readout_modules.items() if m.predicts_flow
    )
    optic_flow_right = right_res["outputs"][decoder_key]
    optic_flow_left = left_res["outputs"][decoder_key]
    optic_flow = torch.cat((optic_flow_right, optic_flow_left), dim=2)

    phi = model.column_phi_theta_two_eyes[0].to(dtype=optic_flow.dtype)
    theta = model.column_phi_theta_two_eyes[1].to(dtype=optic_flow.dtype)

    valid_mask = (
        ~torch.isnan(optic_flow[0, 0, :, 0])
        & ~torch.isnan(optic_flow[0, 0, :, 1])
    )
    if oob is not None:
        valid_mask = valid_mask & ~oob

    d_right = optic_flow[0, :num_frames, :, 0][:, valid_mask]
    d_up = optic_flow[0, :num_frames, :, 1][:, valid_mask]

    with torch.no_grad():
        _, _, rho_valid = model.estimate_ego_motion_from_flow(
            phi[valid_mask], theta[valid_mask], d_right, d_up,
        )

    twoN = optic_flow.shape[2]
    rho_full = torch.full(
        (num_frames, twoN), float("nan"),
        device=rho_valid.device, dtype=rho_valid.dtype,
    )
    rho_full[:, valid_mask] = rho_valid
    return rho_full, valid_mask


def _integrate_pose(omega_deg, T_vec, dt):
    """Integrate body-frame omega (deg/s) and T to (rotvec, position) tracks
    in the world frame.

    Returns dict with:
        rot: (T, 3) cumulative rotation vector in degrees.
        pos: (T, 3) cumulative position in T's units * dt.
    """
    omega_t = torch.tensor(omega_deg, dtype=torch.float64)
    T_t = torch.tensor(T_vec, dtype=torch.float64)
    rot = nm.integrate_angular_velocity(omega_t, dt).numpy()
    pos = nm.integrate_translational_velocity(T_t, omega_t, dt).numpy()
    return {"rot": rot, "pos": pos}


def _integrate_rotmats_pos(omega_deg, T_vec, dt):
    """Integrate body-frame rates to per-frame world-frame rotation matrices
    and positions.

    Returns:
        R_t: (T, 3, 3) numpy array of rotation matrices.
        p_t: (T, 3) numpy array of positions.
    """
    omega = torch.tensor(omega_deg, dtype=torch.float64)
    v = torch.tensor(T_vec, dtype=torch.float64)
    rotvec_rad = omega * (np.pi / 180) * dt
    R_inc = nm._rodrigues(rotvec_rad)  # (T, 3, 3)

    T_len = omega.shape[0]
    R = torch.eye(3, dtype=torch.float64)
    p = torch.zeros(3, dtype=torch.float64)
    R_list = []
    p_list = []
    for t in range(T_len):
        R = R_inc[t] @ R
        v_world = R @ v[t]
        p = p + v_world * dt
        R_list.append(R.clone())
        p_list.append(p.clone())
    return torch.stack(R_list, dim=0).numpy(), torch.stack(p_list, dim=0).numpy()


# ============================================================================
# Photometric warping on the ommatidium grid
# ============================================================================

def _cross(a, b):
    """Broadcasting cross product of two 3-vector tensors along the last dim."""
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


class OmmatidiumWarper:
    """Differentiable resampler of stimulus values on the ommatidium grid.

    Builds a KDTree once on unit-sphere column directions, finds the k
    nearest neighbors of each column, and at evaluation time computes
    Gaussian-RBF weights based on the angular distance from each *warped*
    direction to those (fixed) neighbors. Neighbor identities are not
    re-queried during the warp; this is a small-warp approximation that
    is accurate when the warp magnitude is much less than the kernel
    bandwidth (sigma_deg, default 5 deg).

    Sample is fully differentiable in the warped direction (the only
    non-differentiable step is the one-time KNN lookup).
    """

    def __init__(self, x_columns, k=12, sigma_deg=5.0,
                 device='cpu', dtype=torch.float32):
        """
        Args:
            x_columns: (M, 3) numpy array of unit viewing directions.
            k: number of neighbors per query.
            sigma_deg: Gaussian RBF bandwidth in degrees.
            device, dtype: torch backend for sampling.
        """
        self.M = x_columns.shape[0]
        self.k = k
        self.sigma_rad = float(sigma_deg) * np.pi / 180.0
        self.device = device
        self.dtype = dtype

        tree = cKDTree(x_columns)
        _, idx = tree.query(x_columns, k=k)  # (M, k)
        self.neighbor_idx = torch.tensor(idx, device=device, dtype=torch.long)
        self.neighbor_x = torch.tensor(
            x_columns[idx], device=device, dtype=dtype,
        )  # (M, k, 3)

    def sample(self, L_src, y):
        """Sample stimulus L_src at warped (target) directions y per column.

        Uses a chord-squared distance kernel (no ``acos``) which is finite
        and differentiable everywhere on the sphere -- in particular at
        cos = 1, where ``acos`` has infinite gradient and breaks autograd.

        For small angles, chord^2 = 2(1 - cos) ~ angle^2, so the kernel
        reduces to a standard Gaussian RBF on the angle.

        Args:
            L_src: (..., M) source stimulus values.
            y: (..., M, 3) target directions (will be normalized internally).
        Returns:
            (..., M) sampled values, differentiable in y.
        """
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (y.unsqueeze(-2) * self.neighbor_x).sum(dim=-1)  # (..., M, k)
        chord_sq = (2.0 * (1.0 - cos)).clamp(min=0.0)
        w = torch.exp(-chord_sq / (2.0 * self.sigma_rad ** 2))
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        L_neighbors = L_src[..., self.neighbor_idx]  # (..., M, k)
        return (L_neighbors * w).sum(dim=-1)


def _predict_warped_directions(omega_rad, t_hat, x, rho, dt):
    """Linearized backward-warp direction per column.

    For each column i, the body-frame direction at time t that the column
    will be looking *at* time t+dt:

        y_i = x_i + (omega x x_i) * dt + rho_i * (Pi_i @ t_hat) * dt

    Args:
        omega_rad: (T_pair, 3) angular velocity in rad/s.
        t_hat: (T_pair, 3) unit translation direction.
        x: (M, 3) column viewing directions.
        rho: (T_pair, M) per-column nearness ||T||/Z.
        dt: time step.
    Returns:
        y: (T_pair, M, 3) target directions (un-normalized).
    """
    cross_omx = _cross(omega_rad.unsqueeze(1), x.unsqueeze(0))  # (T_pair, M, 3)
    t_dot_x = (t_hat.unsqueeze(1) * x.unsqueeze(0)).sum(dim=-1, keepdim=True)
    pi_that = t_hat.unsqueeze(1) - t_dot_x * x.unsqueeze(0)     # (T_pair, M, 3)
    return x.unsqueeze(0) + cross_omx * dt + rho.unsqueeze(-1) * pi_that * dt


def _photo_residual(omega, T_param, L_t0, L_t1, x, rho_pair,
                    valid_pair_f, dt, warper):
    """Compute the photometric MSE residual and per-frame residuals.

    Returns scalar mean residual and (T_pair,) per-frame residual.
    """
    deg2rad = np.pi / 180.0
    omega_pair = omega[:-1]
    T_pair = T_param[:-1]
    t_hat = T_pair / (T_pair.norm(dim=-1, keepdim=True) + 1e-12)
    omega_rad = omega_pair * deg2rad

    y = _predict_warped_directions(omega_rad, t_hat, x, rho_pair, dt)
    L_hat = warper.sample(L_t0, y)
    diff = (L_t1 - L_hat) * valid_pair_f
    sq = diff ** 2
    n_per_frame = valid_pair_f.sum(dim=-1)
    per_frame = sq.sum(dim=-1) / (n_per_frame + 1e-12)
    mean_loss = sq.sum() / (valid_pair_f.sum() + 1e-12)
    return mean_loss, per_frame


def photometric_refine(stim, x_2e, rho, valid, omega_init, T_init, dt,
                       num_iters=80, lr=1e-2, smooth=0.0, device='cpu'):
    """Refine per-frame omega and T_direction by photometric matching.

    Solves for {omega_t, T_t}_{t=0..T-2} minimizing the SSE between
    L(t+1) and a warp of L(t) on the ommatidium grid, where the warp
    uses the linearized flow:

        y_i = x_i + (omega x x_i) * dt + rho_i * (Pi_i @ t_hat) * dt

    The depth structure rho is fixed from the model. T is parameterized as
    a 3-vector but only its direction enters the warp; the magnitude is
    unobservable from monocular flow on a single frame pair.

    Args:
        stim: (T, 2N) per-frame stimulus values, both eyes concatenated.
        x_2e: (2N, 3) unit direction per column.
        rho: (T, 2N) per-column nearness rho_i = ||T||/Z (NaN at invalid).
        valid: (2N,) bool mask of always-valid columns.
        omega_init, T_init: (T, 3) initial ego-motion from the model.
        dt: time step (model_dt).
        num_iters: optimizer iterations.
        lr: Adam learning rate.
        smooth: temporal smoothness weight on omega and t_hat.
        device: torch device.
    Returns:
        omega_out: (T, 3) refined angular velocity (deg/s).
        T_out: (T, 3) refined translation: t_hat_phot * ||T_init|| per frame.
        residual_pre: (T-1,) per-frame residual at initialization.
        residual_post: (T-1,) per-frame residual after optimization.
    """
    T_len, M = stim.shape
    dtype = torch.float32

    L = torch.tensor(stim, device=device, dtype=dtype)              # (T, M)
    x = torch.tensor(x_2e, device=device, dtype=dtype)              # (M, 3)
    rho_clean = torch.tensor(
        np.nan_to_num(rho, nan=0.0), device=device, dtype=dtype,
    )                                                                # (T, M)
    rho_finite_np = np.isfinite(rho)
    valid_pf = (
        torch.tensor(valid, device=device, dtype=torch.bool).unsqueeze(0)
        & torch.tensor(rho_finite_np, device=device, dtype=torch.bool)
    )                                                                # (T, M)
    valid_pair_f = (valid_pf[:-1] & valid_pf[1:]).to(dtype=dtype)   # (T-1, M)

    warper = OmmatidiumWarper(
        x_2e, k=12, sigma_deg=5.0, device=device, dtype=dtype,
    )

    omega = torch.tensor(omega_init, device=device, dtype=dtype, requires_grad=True)
    T_param = torch.tensor(T_init, device=device, dtype=dtype, requires_grad=True)

    L_t0 = L[:-1]
    L_t1 = L[1:]
    rho_pair = rho_clean[:-1]

    with torch.no_grad():
        loss_init, residual_pre_t = _photo_residual(
            omega, T_param, L_t0, L_t1, x, rho_pair, valid_pair_f, dt, warper,
        )
    print(f"  initial photo loss: {loss_init.item():.6f}")

    opt = torch.optim.Adam([omega, T_param], lr=lr)
    log_step = max(1, num_iters // 5)
    best_loss = float(loss_init.item())
    best_omega = omega.detach().clone()
    best_T = T_param.detach().clone()
    for it in range(num_iters):
        opt.zero_grad()
        loss, _ = _photo_residual(
            omega, T_param, L_t0, L_t1, x, rho_pair, valid_pair_f, dt, warper,
        )
        if not torch.isfinite(loss):
            print(f"    iter {it+1}: non-finite loss, reverting to best params and halving lr.")
            with torch.no_grad():
                omega.copy_(best_omega)
                T_param.copy_(best_T)
            for g in opt.param_groups:
                g["lr"] *= 0.5
                if g["lr"] < 1e-6:
                    print("    lr too small, stopping.")
                    break
            continue
        total = loss
        if smooth > 0:
            d_omega = omega[1:] - omega[:-1]
            t_hat = T_param / (T_param.norm(dim=-1, keepdim=True) + 1e-8)
            d_that = t_hat[1:] - t_hat[:-1]
            total = total + smooth * ((d_omega ** 2).mean() + (d_that ** 2).mean())
        total.backward()
        torch.nn.utils.clip_grad_norm_([omega, T_param], max_norm=10.0)
        opt.step()
        with torch.no_grad():
            # Re-normalize T_param to keep it on the unit sphere; the
            # magnitude direction is a redundant DOF and removing it
            # avoids ill-conditioned gradients near small ||T||.
            T_param.div_(T_param.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_omega.copy_(omega.detach())
                best_T.copy_(T_param.detach())
        if it == 0 or (it + 1) % log_step == 0 or it == num_iters - 1:
            print(f"    iter {it+1}/{num_iters}: photo loss={loss.item():.6f}")

    with torch.no_grad():
        omega.copy_(best_omega)
        T_param.copy_(best_T)

    with torch.no_grad():
        loss_final, residual_post_t = _photo_residual(
            omega, T_param, L_t0, L_t1, x, rho_pair, valid_pair_f, dt, warper,
        )
    print(f"  final photo loss:   {loss_final.item():.6f}")

    omega_np = omega.detach().cpu().numpy().astype(np.float64)
    T_np = T_param.detach().cpu().numpy().astype(np.float64)
    # Replace T magnitude with the model's per-frame magnitude so that
    # photometric integration uses the same scale as the model integration.
    init_norm = np.linalg.norm(T_init, axis=-1, keepdims=True)
    new_norm = np.linalg.norm(T_np, axis=-1, keepdims=True)
    t_hat_np = T_np / np.maximum(new_norm, 1e-12)
    T_out = t_hat_np * init_norm

    return (
        omega_np,
        T_out,
        residual_pre_t.cpu().numpy().astype(np.float64),
        residual_post_t.cpu().numpy().astype(np.float64),
    )


# ============================================================================
# Top-down depth map
# ============================================================================

def _sample_rho_at_directions(rho_field, x_columns, query_dirs,
                              sigma_deg=5.0, k=12):
    """Resample a (T, M) rho field at A query directions on the unit sphere.

    Uses the same chord-distance Gaussian RBF as OmmatidiumWarper. Invalid
    columns (rho <= 0 or NaN) are excluded from the kernel sum.

    Args:
        rho_field: (T, M) per-frame nearness, with non-positive / NaN
            values marking invalid columns.
        x_columns: (M, 3) unit viewing directions of the columns.
        query_dirs: (A, 3) unit query directions.
        sigma_deg: kernel bandwidth in degrees.
        k: number of nearest columns per query.
    Returns:
        rho_q: (T, A) interpolated rho at each query direction (NaN if no
            valid neighbor).
        coverage_q: (T, A) sum of valid kernel weights at each query (a
            small value indicates the query is poorly covered).
    """
    rho_field = np.asarray(rho_field, dtype=np.float64)
    x_columns = np.asarray(x_columns, dtype=np.float64)
    query_dirs = np.asarray(query_dirs, dtype=np.float64)

    tree = cKDTree(x_columns)
    _, idx = tree.query(query_dirs, k=k)              # (A, k)
    neighbor_x = x_columns[idx]                        # (A, k, 3)
    cos = (query_dirs[:, None, :] * neighbor_x).sum(axis=-1)  # (A, k)
    chord_sq = np.maximum(2.0 * (1.0 - cos), 0.0)
    sigma_rad = sigma_deg * np.pi / 180.0
    w_kernel = np.exp(-chord_sq / (2.0 * sigma_rad ** 2))      # (A, k)

    rho_neighbors = rho_field[:, idx]                  # (T, A, k)
    valid_neighbor = np.isfinite(rho_neighbors) & (rho_neighbors > 0)
    eff_w = w_kernel[None, :, :] * valid_neighbor      # (T, A, k)
    coverage_q = eff_w.sum(axis=-1)                    # (T, A)
    rho_w_sum = (eff_w * np.where(valid_neighbor, rho_neighbors, 0.0)).sum(axis=-1)
    rho_q = np.where(coverage_q > 0,
                     rho_w_sum / np.maximum(coverage_q, 1e-12),
                     np.nan)
    return rho_q, coverage_q


def build_room_map(rho, x_2e, valid, omega, T, dt,
                   n_azimuth=360, sigma_deg=5.0, k=12,
                   grid_size=200, depth_percentile=98.0):
    """Build a 2D LIDAR-style room map by resampling depth in 1-degree
    azimuth bins around the drone, projected to world coords and
    accumulated over the entire flight.

    For each frame t and each body-frame horizontal azimuth psi (0..360 at
    n_azimuth bins, theta = pi/2), interpolate rho_psi from the model's
    per-column nearness via a chord-distance Gaussian RBF. The depth in
    that direction is Z_psi = ||T_t|| / rho_psi. The world XY of the
    obstacle in that direction is

        world_xy(t, psi) = p_t[:2] + Z_psi * (R_t @ horizontal(psi))[:2]

    where horizontal(psi) = (cos psi, sin psi, 0). Bins where rho is
    missing (no valid neighbor) are dropped.

    Per-measurement Fisher weight (inverse-variance on nearness):

        w(t, psi) = ||T_t||^2 - (T_t . horizontal_psi_world)^2
                  = ||Pi_psi T_t||^2.

    Returns dict with raw scatter arrays (one row per accepted
    (frame, azimuth) sample), drone trajectory, and an optional 2D
    grid of weighted-mean nearness for downstream use.
    """
    T_len, M = rho.shape
    R_t, p_t = _integrate_rotmats_pos(omega, T, dt)         # (T, 3, 3), (T, 3)
    T_mag = np.linalg.norm(T, axis=-1)                       # (T,)

    psi = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    body_dirs = np.stack(
        [np.cos(psi), np.sin(psi), np.zeros_like(psi)], axis=-1,
    )                                                        # (A, 3)

    rho_valid = valid[None, :] & np.isfinite(rho) & (rho > 0)
    rho_field = np.where(rho_valid, rho, np.nan)

    rho_q, coverage_q = _sample_rho_at_directions(
        rho_field, x_2e, body_dirs, sigma_deg=sigma_deg, k=k,
    )                                                        # (T, A)

    body_world_dirs = np.einsum('tij,aj->tai', R_t, body_dirs)  # (T, A, 3)
    Z_q = T_mag[:, None] / np.maximum(rho_q, 1e-12)            # (T, A)
    world_pts = body_world_dirs * Z_q[..., None] + p_t[:, None, :]
    world_xy = world_pts[..., :2]                              # (T, A, 2)

    # Fisher weight: ||T||^2 - (T . body_world_dir)^2. Equivalent to
    # ||T||^2 - (T . body_dir)^2 since rotation preserves inner products,
    # but compute in body frame for clarity.
    T_dot_body = T @ body_dirs.T                              # (T, A)
    fisher_w = np.maximum(T_mag[:, None] ** 2 - T_dot_body ** 2, 0.0)

    valid_q = (
        np.isfinite(rho_q) & (rho_q > 0) & (coverage_q > 1e-6)
        & (T_mag[:, None] > 1e-9)
    )

    flat_xy = world_xy.reshape(-1, 2)
    flat_rho = rho_q.reshape(-1)
    flat_Z = Z_q.reshape(-1)
    flat_w = fisher_w.reshape(-1)
    flat_frame = np.repeat(np.arange(T_len), n_azimuth)
    flat_psi = np.tile(psi, T_len)
    keep = valid_q.reshape(-1)
    flat_xy = flat_xy[keep]
    flat_rho = flat_rho[keep]
    flat_Z = flat_Z[keep]
    flat_w = flat_w[keep]
    flat_frame = flat_frame[keep]
    flat_psi = flat_psi[keep]

    traj_xy = p_t[:, :2]
    if flat_xy.shape[0] == 0:
        return {
            "scatter_xy": np.zeros((0, 2)),
            "scatter_rho": np.zeros((0,)),
            "scatter_Z": np.zeros((0,)),
            "scatter_weight": np.zeros((0,)),
            "scatter_frame": np.zeros((0,), dtype=np.int64),
            "scatter_azimuth": np.zeros((0,)),
            "trajectory_xy": traj_xy,
            "grid_x_edges": np.array([0.0, 1.0]),
            "grid_y_edges": np.array([0.0, 1.0]),
            "grid_weight": np.zeros((grid_size, grid_size)),
            "grid_nearness": np.full((grid_size, grid_size), np.nan),
        }

    xy_lo = np.percentile(flat_xy, 100 - depth_percentile, axis=0)
    xy_hi = np.percentile(flat_xy, depth_percentile, axis=0)
    if traj_xy.size > 0:
        xy_lo = np.minimum(xy_lo, traj_xy.min(axis=0))
        xy_hi = np.maximum(xy_hi, traj_xy.max(axis=0))
    pad = 0.05 * np.maximum(xy_hi - xy_lo, 1e-6)
    xy_lo = xy_lo - pad
    xy_hi = xy_hi + pad

    x_edges = np.linspace(xy_lo[0], xy_hi[0], grid_size + 1)
    y_edges = np.linspace(xy_lo[1], xy_hi[1], grid_size + 1)

    in_box = (
        (flat_xy[:, 0] >= xy_lo[0]) & (flat_xy[:, 0] <= xy_hi[0])
        & (flat_xy[:, 1] >= xy_lo[1]) & (flat_xy[:, 1] <= xy_hi[1])
    )
    box_xy = flat_xy[in_box]
    box_rho = flat_rho[in_box]
    box_w = flat_w[in_box]
    weight_sum, _, _ = np.histogram2d(
        box_xy[:, 0], box_xy[:, 1], bins=[x_edges, y_edges], weights=box_w,
    )
    rho_w_sum, _, _ = np.histogram2d(
        box_xy[:, 0], box_xy[:, 1], bins=[x_edges, y_edges],
        weights=box_w * box_rho,
    )
    nearness_grid = np.divide(
        rho_w_sum, weight_sum,
        out=np.full_like(rho_w_sum, np.nan, dtype=np.float64),
        where=weight_sum > 0,
    )

    return {
        "scatter_xy": flat_xy,
        "scatter_rho": flat_rho,
        "scatter_Z": flat_Z,
        "scatter_weight": flat_w,
        "scatter_frame": flat_frame,
        "scatter_azimuth": flat_psi,
        "trajectory_xy": traj_xy,
        "grid_x_edges": x_edges,
        "grid_y_edges": y_edges,
        "grid_weight": weight_sum,
        "grid_nearness": nearness_grid,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_topdown_map(grid, pose_A, pose_B, pose_C, output_path,
                     clip_idx, clip_id):
    """Single-panel top-down room map.

    Each accepted (frame, azimuth) depth sample is drawn as a single
    scatter point at its world XY, with low alpha so the room shape
    emerges from where many samples pile up across the flight. The
    drone trajectory (Pose B) is overlaid as a line.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pts_xy = grid["scatter_xy"]
    pts_w = grid["scatter_weight"]
    pts_frame = grid["scatter_frame"]
    traj_xy = grid["trajectory_xy"]
    x_edges = grid["grid_x_edges"]
    y_edges = grid["grid_y_edges"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    if pts_xy.shape[0] > 0:
        # Drop the lowest-weight tail (low-translation observations are
        # nearly uninformative). Keep alpha low so dense regions stand out.
        if pts_w.size > 100:
            w_thresh = np.percentile(pts_w, 25.0)
            keep = pts_w >= w_thresh
            xy = pts_xy[keep]
            color_idx = pts_frame[keep]
        else:
            xy = pts_xy
            color_idx = pts_frame
        n = xy.shape[0]
        # Alpha scaled so heavily-overlapped regions go opaque; sparse
        # outliers stay faint.
        alpha = float(np.clip(2000.0 / max(n, 1), 0.02, 0.4))
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=color_idx, cmap="viridis",
            s=2, alpha=alpha, linewidths=0, rasterized=True,
        )

    if traj_xy.shape[0] > 0:
        ax.plot(traj_xy[:, 0], traj_xy[:, 1],
                color="white", linewidth=2.5, zorder=4)
        ax.plot(traj_xy[:, 0], traj_xy[:, 1],
                color="black", linewidth=1.2, zorder=5,
                label="drone trajectory (Pose B)")
        ax.scatter([traj_xy[0, 0]], [traj_xy[0, 1]],
                   color="lime", s=70, zorder=6, edgecolors="black",
                   label="start")
        ax.scatter([traj_xy[-1, 0]], [traj_xy[-1, 1]],
                   color="red", s=70, zorder=6, edgecolors="black",
                   label="end")

    if (np.all(np.isfinite(x_edges)) and np.all(np.isfinite(y_edges))):
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("X (m, world forward at start)")
    ax.set_ylabel("Y (m, world left at start)")
    ax.set_title(
        f"Estimated room shape and drone trajectory — clip {clip_idx} "
        f"(id={clip_id})\n"
        f"point = depth estimate at one azimuth bin, color = frame index"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved top-down map -> {output_path}")


def plot_diagnostics(omega_model, omega_phot, T_model, T_phot,
                     residual_pre, residual_post, model_dt, output_path,
                     clip_idx, clip_id):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    T_len = omega_model.shape[0]
    times = np.arange(T_len) * model_dt
    rot_labels = ["Roll", "Pitch", "Yaw"]
    trans_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(3):
        ax = axes[0, i]
        ax.plot(times, omega_model[:, i], label="model", linewidth=1.2)
        ax.plot(times, omega_phot[:, i], label="photo", linewidth=1.2,
                linestyle="--")
        ax.set_title(f"omega {rot_labels[i]} (deg/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    norm_model = np.linalg.norm(T_model, axis=-1, keepdims=True)
    norm_phot = np.linalg.norm(T_phot, axis=-1, keepdims=True)
    that_model = T_model / np.maximum(norm_model, 1e-12)
    that_phot = T_phot / np.maximum(norm_phot, 1e-12)
    for i in range(3):
        ax = axes[1, i]
        ax.plot(times, that_model[:, i], label="model t_hat", linewidth=1.2)
        ax.plot(times, that_phot[:, i], label="photo t_hat", linewidth=1.2,
                linestyle="--")
        ax.set_title(f"t_hat {trans_labels[i]}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    pair_times = times[:-1]
    axes[2, 0].plot(pair_times, residual_pre, label="pre-refine", linewidth=1.2)
    axes[2, 0].plot(pair_times, residual_post, label="post-refine", linewidth=1.2,
                    linestyle="--")
    axes[2, 0].set_title("Per-frame photometric residual (MSE)")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=8)

    cos_omega = (
        (omega_model * omega_phot).sum(axis=-1)
        / (np.linalg.norm(omega_model, axis=-1)
           * np.linalg.norm(omega_phot, axis=-1) + 1e-12)
    )
    cos_that = (that_model * that_phot).sum(axis=-1)
    axes[2, 1].plot(times, cos_omega, label="omega", linewidth=1.2)
    axes[2, 1].plot(times, cos_that, label="t_hat", linewidth=1.2, linestyle="--")
    axes[2, 1].set_title("Model vs photometric cosine similarity")
    axes[2, 1].set_xlabel("time (s)")
    axes[2, 1].set_ylim(-1.05, 1.05)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(fontsize=8)

    axes[2, 2].axis("off")
    summary = (
        f"Mean residual (pre):  {residual_pre.mean():.6f}\n"
        f"Mean residual (post): {residual_post.mean():.6f}\n"
        f"Mean cos(omega):      {np.nanmean(cos_omega):.4f}\n"
        f"Mean cos(t_hat):      {np.nanmean(cos_that):.4f}\n"
    )
    axes[2, 2].text(0.05, 0.55, summary, family="monospace", fontsize=11,
                    verticalalignment="center")

    fig.suptitle(f"Photometric refinement diagnostics — "
                 f"clip {clip_idx} (id={clip_id})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved diagnostics -> {output_path}")


if __name__ == "__main__":
    main()
