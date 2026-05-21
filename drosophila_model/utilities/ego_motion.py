"""Ego-motion math: flow -> 6-DOF solver, SO(3) rotations, velocity integration,
and the forward direction (synthetic optic flow from ego-motion).

Pure-tensor helpers with one disk read (``load_viewing_directions``). Consumed
by :class:`regions.OpticLobeRegion` (the only region with optic flow), the
ego-motion test suite, ``stimulus_utilities`` (for ground-truth flow), and
analysis / movie scripts that need to integrate target trajectories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Flow -> ego-motion solver
# ---------------------------------------------------------------------------


def _cross_matrix(a):
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    O = torch.zeros_like(ax)
    return torch.stack([
        torch.stack([   O, -az,  ay], dim=-1),
        torch.stack([  az,   O, -ax], dim=-1),
        torch.stack([ -ay,  ax,   O], dim=-1),
    ], dim=-2)


def _spherical_basis(phi, theta):
    x = torch.stack((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ), dim=1)
    e_theta = torch.stack((
        torch.cos(theta) * torch.cos(phi),
        torch.cos(theta) * torch.sin(phi),
        -torch.sin(theta),
    ), dim=1)
    e_phi = torch.stack((
        -torch.sin(phi),
        torch.cos(phi),
        torch.zeros_like(phi),
    ), dim=1)
    return x, e_phi, e_theta


def _estimate_rotation_from_flow(phi, theta, d_right, d_up):
    x, e_phi, e_theta = _spherical_basis(phi, theta)
    u = -d_right[..., None] * e_phi - d_up[..., None] * e_theta
    T_len, N = d_right.shape
    A = _cross_matrix(x)
    A_flat = A.reshape(3 * N, 3)
    u_flat = u.reshape(T_len, 3 * N).T
    AtA = A_flat.T @ A_flat
    Atb = A_flat.T @ u_flat
    omega_star = torch.linalg.solve(AtA, Atb).T
    return omega_star


def _estimate_translation_from_flow(phi, theta, d_right, d_up):
    x, e_phi, e_theta = _spherical_basis(phi, theta)
    v = -d_right[..., None] * e_phi - d_up[..., None] * e_theta
    T_len, N = d_right.shape

    outer = x.unsqueeze(2) @ x.unsqueeze(1)
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(N, 3, 3)
    P = I - outer

    vx = _cross_matrix(v)
    C_blocks = vx @ P
    C = C_blocks.reshape(T_len, 3 * N, 3)
    _, _, Vh = torch.linalg.svd(C, full_matrices=False)
    T_dir = Vh[:, -1]

    w_i = torch.einsum('nij,tj->tni', P, T_dir)
    v_dot_w = (v * w_i).sum(dim=-1)
    w_norm_sq = (w_i * w_i).sum(dim=-1)
    weighted_sign = (v_dot_w * w_norm_sq).sum(dim=-1)
    sign_flip = torch.where(
        weighted_sign > 0,
        torch.tensor(-1.0, device=x.device, dtype=d_right.dtype),
        torch.tensor(1.0, device=x.device, dtype=d_right.dtype),
    )
    T_dir = T_dir * sign_flip[:, None]
    v_dot_w = v_dot_w * sign_flip[:, None]

    rho_star = -v_dot_w / (w_norm_sq + 1e-12)

    focus_threshold = 0.1
    valid = w_norm_sq > focus_threshold
    rho_masked = torch.where(valid, rho_star, torch.full_like(rho_star, float('nan')))
    rho_agg = torch.nanmedian(rho_masked, dim=-1).values
    rho_agg = torch.where(
        torch.isnan(rho_agg),
        torch.nanmedian(rho_star, dim=-1).values,
        rho_agg,
    )
    T_star = rho_agg[:, None] * T_dir
    return T_star, rho_star


def estimate_from_flow(phi, theta, d_right, d_up, terms=('rotation', 'trans')):
    """Estimate 6-DOF ego motion from optic flow on a sphere.

    Coordinate system:
        +X: Forward (phi=0, theta=pi/2)
        +Y: Left    (phi=pi/2, theta=pi/2)
        +Z: Up      (theta=0)

    ``terms`` selects which components to solve for. With both, rotation is
    estimated first (linear least-squares), then subtracted from the flow
    before solving for translation direction via SVD; the per-point
    inverse-depth ``rho = ||T||/Z`` is recovered as a by-product.

    Returns ``(omega_star, T_star, rho_star)``. Inputs may be unbatched
    ``(N,)`` or batched ``(T, N)``; the returned shapes follow.
    """
    terms = tuple(terms)
    valid_terms = {'rotation', 'trans'}
    if not terms or not set(terms).issubset(valid_terms):
        raise ValueError(
            f"`terms` must be a non-empty subset of {valid_terms}, got {terms}"
        )

    unbatched = d_right.dim() == 1
    if unbatched:
        d_right = d_right.unsqueeze(0)
        d_up = d_up.unsqueeze(0)

    T_len, N = d_right.shape
    device, dtype = d_right.device, d_right.dtype

    if 'rotation' in terms:
        omega_star = _estimate_rotation_from_flow(phi, theta, d_right, d_up)
    else:
        omega_star = torch.zeros(T_len, 3, device=device, dtype=dtype)

    if 'trans' in terms:
        if 'rotation' in terms:
            x, e_phi, e_theta = _spherical_basis(phi, theta)
            A = _cross_matrix(x)
            u_rot = torch.einsum('nij,tj->tni', A, omega_star)
            d_right_rot = -(u_rot * e_phi).sum(dim=-1)
            d_up_rot = -(u_rot * e_theta).sum(dim=-1)
            d_right_t = d_right - d_right_rot
            d_up_t = d_up - d_up_rot
        else:
            d_right_t = d_right
            d_up_t = d_up
        T_star, rho_star = _estimate_translation_from_flow(
            phi, theta, d_right_t, d_up_t,
        )
    else:
        T_star = torch.zeros(T_len, 3, device=device, dtype=dtype)
        rho_star = torch.zeros(T_len, N, device=device, dtype=dtype)

    if unbatched:
        omega_star = omega_star.squeeze(0)
        T_star = T_star.squeeze(0)
        rho_star = rho_star.squeeze(0)

    return omega_star, T_star, rho_star


# ---------------------------------------------------------------------------
# SO(3) rotations
# ---------------------------------------------------------------------------


def _rodrigues(rotvec):
    """Rotation vectors (axis * angle, radians) to rotation matrices.

    Uses Taylor expansions for ``sin(theta)/theta`` and ``(1-cos(theta))/theta^2``
    at small angles to avoid catastrophic cancellation in the backward pass.
    """
    theta_sq = (rotvec * rotvec).sum(dim=-1, keepdim=True)

    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    O = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)

    small = (theta_sq < 1e-6).squeeze(-1)
    safe_sq = theta_sq.clamp(min=1e-24)
    theta = safe_sq.sqrt()

    sinc = torch.where(small.unsqueeze(-1), 1 - theta_sq / 6, torch.sin(theta) / theta)
    cosc = torch.where(small.unsqueeze(-1), 0.5 - theta_sq / 24, (1 - torch.cos(theta)) / safe_sq)

    sinc = sinc.unsqueeze(-1)
    cosc = cosc.unsqueeze(-1)
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return I + sinc * K + cosc * (K @ K)


def _rotmat_to_rotvec(R):
    """Rotation matrices to rotation vectors (axis * angle, radians)."""
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)

    sin_angle = (v * v).sum(dim=-1).clamp(min=1e-24).sqrt() / 2
    cos_angle = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2

    angle = torch.atan2(sin_angle, cos_angle)

    angle_sq = angle * angle
    scale_small = 0.5 + angle_sq / 12
    scale_large = angle / (2 * sin_angle)
    scale = torch.where(sin_angle.abs() < 1e-4, scale_small, scale_large)

    return v * scale.unsqueeze(-1)


def integrate_angular_velocity(omega_deg, dt):
    """Integrate angular velocity to cumulative orientation via SO(3) composition."""
    squeeze = omega_deg.dim() == 2
    if squeeze:
        omega_deg = omega_deg.unsqueeze(0)

    rotvec_rad = omega_deg * (np.pi / 180) * dt
    R_inc = _rodrigues(rotvec_rad)

    B, T = omega_deg.shape[:2]
    I = torch.eye(3, device=omega_deg.device, dtype=omega_deg.dtype)
    R = I.unsqueeze(0).expand(B, -1, -1).clone()
    rotvecs = []
    for t in range(T):
        R = R_inc[:, t] @ R
        rotvecs.append(_rotmat_to_rotvec(R))
    result = torch.stack(rotvecs, dim=1) * (180 / np.pi)

    if squeeze:
        result = result.squeeze(0)
    return result


def integrate_translational_velocity(v, omega_deg, dt):
    """Integrate body-frame translational velocity to world-frame position."""
    squeeze = v.dim() == 2
    if squeeze:
        v = v.unsqueeze(0)
        omega_deg = omega_deg.unsqueeze(0)

    rotvec_rad = omega_deg * (np.pi / 180) * dt
    R_inc = _rodrigues(rotvec_rad)

    B, T = v.shape[:2]
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I.unsqueeze(0).expand(B, -1, -1).clone()
    pos = torch.zeros(B, 3, device=v.device, dtype=v.dtype)
    positions = []
    for t in range(T):
        R = R_inc[:, t] @ R
        v_world = (R @ v[:, t, :, None]).squeeze(-1)
        pos = pos + v_world * dt
        positions.append(pos.clone())
    result = torch.stack(positions, dim=1)

    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Synthetic optic-flow generation (forward direction of estimate_from_flow)
# ---------------------------------------------------------------------------


def load_viewing_directions(connectome_folder: str | Path | None = None):
    """Load optic-lobe column phi/theta and stack both hemispheres.

    Args:
        connectome_folder: Path to the connectome bundle. When ``None``,
            defaults to ``../connectome_data`` relative to this file.

    Returns:
        phi, theta: ``(2*N_columns,)`` tensors, with the second half
        mirrored across the midline (``-phi``).
    """
    if connectome_folder is None:
        connectome_folder = Path(__file__).parent.parent / 'connectome_data'
    connectome_folder = Path(connectome_folder)
    column_info_df = pd.read_parquet(
        connectome_folder / 'area_extras' / 'optic_lobe_columns.parquet'
    )

    theta_half = torch.tensor(column_info_df['theta'].values, dtype=torch.float32)
    phi_half = torch.tensor(column_info_df['phi'].values, dtype=torch.float32)

    theta = torch.cat([theta_half, theta_half])
    phi = torch.cat([phi_half, -phi_half])
    return phi, theta


def generate_optic_flow(phi, theta, omega, T, Z):
    """Forward optic-flow model: rotation + translation + depth -> angular flow.

    Inverse of :func:`estimate_from_flow`. Same spherical basis (see
    :func:`_spherical_basis`) and same sign conventions, so a flow generated
    here and fed back into the estimator recovers the original ``(omega, T)``.

    Args:
        phi: ``(N,)`` azimuthal angles.
        theta: ``(N,)`` polar angles.
        omega: ``(3,)`` or ``(T, 3)`` body-frame angular velocity vector(s).
        T: ``(3,)`` or ``(T, 3)`` body-frame translational velocity vector(s).
        Z: ``(N,)`` or scalar per-column depths.

    Returns:
        u_right, u_up: angular flow on the sphere. ``(N,)`` for single-frame
        ``(omega, T)``, ``(T, N)`` for time-varying.
    """
    if omega.dim() != T.dim():
        raise ValueError(
            f"omega and T must have the same number of dimensions, "
            f"got omega: {omega.shape} and T: {T.shape}"
        )
    single_timepoint = omega.dim() == 1
    if single_timepoint:
        omega = omega.unsqueeze(0)
        T = T.unsqueeze(0)

    x, e_phi, e_theta = _spherical_basis(phi, theta)
    N = x.shape[0]

    # Rotation contribution: u = -Ω × x = [x]_× Ω
    A = _cross_matrix(x)
    u_rot = torch.einsum('nij,tj->tni', A, omega)

    # Translation contribution: u_trans = -(1/Z) * Π * T,  Π = I - x xᵀ
    outer = x.unsqueeze(2) @ x.unsqueeze(1)
    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(N, 3, 3)
    Pi = I - outer
    Pi_T = torch.einsum('nij,tj->tni', Pi, T)
    if torch.is_tensor(Z) and Z.dim() > 0:
        u_trans = -Pi_T / Z.unsqueeze(0).unsqueeze(-1)
    else:
        u_trans = -Pi_T / Z

    u = u_rot + u_trans

    # Project onto spherical basis (sign matches estimate_from_flow).
    u_right = -(u * e_phi).sum(dim=-1)
    u_up = -(u * e_theta).sum(dim=-1)

    if single_timepoint:
        u_right = u_right.squeeze(0)
        u_up = u_up.squeeze(0)

    return u_right, u_up
