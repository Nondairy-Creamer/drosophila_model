"""Ego-motion math: flow -> 6-DOF solver, SO(3) rotations, velocity integration,
and the forward direction (synthetic optic flow from ego-motion).

Pure-tensor helpers with one disk read (``load_viewing_directions``). Consumed
by :class:`regions.OpticLobeRegion` (the only region with optic flow), the
ego-motion test suite, ``data_handling.stimulus_generators`` (for ground-truth
flow), and analysis / movie scripts that need to integrate target trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Flow -> ego-motion solver
# ---------------------------------------------------------------------------


@dataclass
class EyeFlow:
    """Optic flow from one eye, plus where that eye sits on the body.

    All angles/directions are in the shared body frame (+x forward, +y left,
    +z up); each eye supplies its own ``phi``/``theta`` columns. ``offset`` is
    the vector from the body center of mass to the eye, in the same body frame
    and the same length units as the desired translation ``T``. A zero offset
    recovers the single-center (monocular) model.

    Shapes: ``phi``/``theta`` are ``(N,)``; ``d_right``/``d_up`` are ``(N,)``
    (single frame) or ``(T, N)`` (batched); ``offset`` is ``(3,)``.
    """

    phi: torch.Tensor
    theta: torch.Tensor
    d_right: torch.Tensor
    d_up: torch.Tensor
    offset: torch.Tensor | None = None

    def offset_vec(self, device, dtype):
        if self.offset is None:
            return torch.zeros(3, device=device, dtype=dtype)
        return self.offset.to(device=device, dtype=dtype)


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


def _derotate_flow(phi, theta, d_right, d_up, omega_star):
    """Subtract the rotation-induced flow, leaving translation-only flow.

    ``v_i = u_i - [d_i]_x omega`` re-expressed in the (right, up) retinal
    components. Depth-independent, so it works point-by-point regardless of
    which eye the column belongs to.
    """
    x, e_phi, e_theta = _spherical_basis(phi, theta)
    A = _cross_matrix(x)
    u_rot = torch.einsum('nij,tj->tni', A, omega_star)
    d_right_rot = -(u_rot * e_phi).sum(dim=-1)
    d_up_rot = -(u_rot * e_theta).sum(dim=-1)
    return d_right - d_right_rot, d_up - d_up_rot


# Reciprocal condition number (smallest/largest eigenvalue) of the joint
# translation normal matrix below which the rotation-induced parallax is too
# weak to trust absolute scale; such timesteps fall back to a direction-only
# (relative-scale) estimate. True degeneracy (no rotation) sits near float
# zero; clear rotation gives ratios >~1e-2. Raise this if real, noisy data
# yields unstable absolute scale at weak parallax.
_SCALE_COND_THRESHOLD = 1e-4


def estimate_from_flow(eyes, terms=('rotation', 'trans')):
    """Estimate 6-DOF ego motion from optic flow seen by one or more eyes.

    Coordinate system (shared body frame):
        +X: Forward (phi=0, theta=pi/2)
        +Y: Left    (phi=pi/2, theta=pi/2)
        +Z: Up      (theta=0)

    Each :class:`EyeFlow` carries its own viewing columns, retinal flow, and an
    ``offset`` from the body center of mass. Body rotation ``Omega`` gives an
    eye at offset ``r`` an extra linear velocity ``Omega x r``, so the eye's
    effective translation is ``T_e = T + Omega x r``; the flow model per column
    is ``u_i = -Omega x d_i - (1/Z_i) Pi_i T_e``.

    Pipeline:
      1. Rotation: stacked linear least-squares over all eyes' columns. The
         rotation term is the same shared ``Omega`` and is offset-independent.
      2. De-rotate every column. Each column gives the colinearity constraint
         ``[v_i]_x Pi_i (T + b_e) = 0``, i.e. ``C_i T = -C_i b_e`` with
         ``b_e = Omega x r_e`` known in absolute units.
      3. Translation: solve ``min_T || C T - d ||^2`` jointly over every column
         of every eye in one least-squares (not per-eye, then merged). The
         known inter-eye parallax makes ``d`` non-zero and the system full-rank,
         so absolute scale is recovered. A second pass drops focus-of-expansion
         columns. Per-point ``rho_i = ||T_e||/Z_i`` falls out of the final ``T``.

    With a single eye (or when the parallax ``Omega x (r_L - r_R)`` is too weak
    to condition the scale system) absolute scale is unrecoverable and ``T`` is
    returned up to the usual positive multiplier.

    ``terms`` selects which components to solve for.

    Returns ``(omega_star, T_star, rho_star)`` where ``rho_star`` concatenates
    the per-eye per-point inverse depths in eye order. Per-eye inputs may be
    unbatched ``(N,)`` or batched ``(T, N)``; the returned shapes follow.
    """
    terms = tuple(terms)
    valid_terms = {'rotation', 'trans'}
    if not terms or not set(terms).issubset(valid_terms):
        raise ValueError(
            f"`terms` must be a non-empty subset of {valid_terms}, got {terms}"
        )

    if isinstance(eyes, EyeFlow):
        eyes = [eyes]
    eyes = list(eyes)
    if not eyes:
        raise ValueError("`eyes` must contain at least one EyeFlow")

    unbatched = eyes[0].d_right.dim() == 1
    d_rights = [e.d_right.unsqueeze(0) if unbatched else e.d_right for e in eyes]
    d_ups = [e.d_up.unsqueeze(0) if unbatched else e.d_up for e in eyes]

    T_len = d_rights[0].shape[0]
    device, dtype = d_rights[0].device, d_rights[0].dtype
    N_per_eye = [d.shape[1] for d in d_rights]

    # --- rotation: shared across eyes, estimated from all columns at once ---
    if 'rotation' in terms:
        phi_all = torch.cat([e.phi for e in eyes])
        theta_all = torch.cat([e.theta for e in eyes])
        d_right_all = torch.cat(d_rights, dim=1)
        d_up_all = torch.cat(d_ups, dim=1)
        omega_star = _estimate_rotation_from_flow(
            phi_all, theta_all, d_right_all, d_up_all,
        )
    else:
        omega_star = torch.zeros(T_len, 3, device=device, dtype=dtype)

    if 'trans' not in terms:
        T_star = torch.zeros(T_len, 3, device=device, dtype=dtype)
        rho_star = torch.zeros(T_len, sum(N_per_eye), device=device, dtype=dtype)
        if unbatched:
            omega_star, T_star, rho_star = (
                omega_star.squeeze(0), T_star.squeeze(0), rho_star.squeeze(0),
            )
        return omega_star, T_star, rho_star

    # --- de-rotate each eye's flow (leaves translation-only flow) ---
    derot = []
    for eye, d_right, d_up in zip(eyes, d_rights, d_ups):
        if 'rotation' in terms:
            d_right_t, d_up_t = _derotate_flow(
                eye.phi, eye.theta, d_right, d_up, omega_star,
            )
        else:
            d_right_t, d_up_t = d_right, d_up
        derot.append((eye, d_right_t, d_up_t))

    if len(eyes) == 1:
        # Monocular: a single sphere has no parallax baseline, so translation is
        # recoverable only up to scale (direction via SVD, magnitude via rho).
        eye, d_right_t, d_up_t = derot[0]
        T_star, rho_star = _estimate_translation_from_flow(
            eye.phi, eye.theta, d_right_t, d_up_t,
        )
    else:
        T_star, rho_star = _estimate_translation_two_eyes(derot, omega_star)

    if unbatched:
        omega_star = omega_star.squeeze(0)
        T_star = T_star.squeeze(0)
        rho_star = rho_star.squeeze(0)

    return omega_star, T_star, rho_star


def estimate_ego_motion_known_depth(phi, theta, d_right, d_up, Z, valid=None):
    """Closed-form 6-DOF ego-motion from optic flow when per-column depth is known.

    Companion to :func:`estimate_from_flow` for the case (e.g. rendered
    ground-truth flow) where the depth ``Z`` of every viewing column is known.
    The forward model ``u_i = x_i x Omega - (1/Z_i) Pi_i T`` (see
    :func:`generate_optic_flow`) is linear in the stacked 6-vector
    ``[Omega; T]``, so a single least-squares solve over all columns recovers
    both with no parallax/scale ambiguity (unlike the depth-free estimator).

    Angular units follow the flow: pass flow in rad/s to get ``Omega`` in rad/s
    and ``T`` in (depth-unit)/s. Columns with non-finite ``Z`` keep only their
    rotation constraint (their translation weight ``1/Z`` is zero), the correct
    limit for infinitely distant points.

    Args:
        phi, theta: ``(N,)`` viewing directions (shared body frame).
        d_right, d_up: ``(N,)`` or ``(T, N)`` retinal flow (right/up components).
        Z: ``(N,)`` or ``(T, N)`` per-column depth (length units of ``T``).
        valid: optional bool mask, ``(N,)`` or ``(T, N)``; ``False`` columns are
            dropped from the solve (e.g. capped/NaN flow).

    Returns:
        ``(omega, T)``, each ``(3,)`` (unbatched) or ``(T, 3)`` (batched).
    """
    unbatched = d_right.dim() == 1
    if unbatched:
        d_right = d_right.unsqueeze(0)
        d_up = d_up.unsqueeze(0)
    T_len, N = d_right.shape
    device, dtype = d_right.device, d_right.dtype

    phi = phi.to(device=device, dtype=dtype)
    theta = theta.to(device=device, dtype=dtype)
    Z = torch.as_tensor(Z, device=device, dtype=dtype)
    if Z.dim() == 1:
        Z = Z.unsqueeze(0).expand(T_len, N)

    x, e_phi, e_theta = _spherical_basis(phi, theta)
    A = _cross_matrix(x)                                   # (N, 3, 3)
    I = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    Pi = I - x.unsqueeze(2) @ x.unsqueeze(1)               # (N, 3, 3)

    # d_right = -(u . e_phi), d_up = -(u . e_theta) with u = A Omega - (1/Z) Pi T.
    # Rotation coefficients are time-independent; translation scales with 1/Z.
    c_right_omega = -torch.einsum('ni,nij->nj', e_phi, A)  # (N, 3)
    c_up_omega = -torch.einsum('ni,nij->nj', e_theta, A)
    ePi_right = torch.einsum('ni,nij->nj', e_phi, Pi)      # (N, 3)
    ePi_up = torch.einsum('ni,nij->nj', e_theta, Pi)

    inv_Z = torch.where(torch.isfinite(Z), 1.0 / Z, torch.zeros_like(Z))
    c_right_T = ePi_right.unsqueeze(0) * inv_Z.unsqueeze(-1)  # (T, N, 3)
    c_up_T = ePi_up.unsqueeze(0) * inv_Z.unsqueeze(-1)

    M_right = torch.cat([c_right_omega.unsqueeze(0).expand(T_len, N, 3), c_right_T], dim=-1)
    M_up = torch.cat([c_up_omega.unsqueeze(0).expand(T_len, N, 3), c_up_T], dim=-1)
    M = torch.cat([M_right, M_up], dim=1)                  # (T, 2N, 6)
    b = torch.cat([d_right, d_up], dim=1)                  # (T, 2N)

    if valid is not None:
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand(T_len, N)
        keep = torch.cat([valid, valid], dim=1).to(dtype)
        M = M * keep.unsqueeze(-1)
        b = b * keep

    sol = torch.linalg.lstsq(M, b.unsqueeze(-1)).solution[..., 0]  # (T, 6)
    omega, T = sol[..., :3], sol[..., 3:]
    if unbatched:
        omega, T = omega.squeeze(0), T.squeeze(0)
    return omega, T


def _flow_constraint(phi, theta, d_right_t, d_up_t):
    """Per-column colinearity blocks for the joint translation solve.

    Returns ``x`` (viewing directions, ``(N,3)``), ``v`` (de-rotated 3D flow,
    ``(T,N,3)``) and ``C_i = [v_i]_x Pi_i`` (``(T,N,3,3)``), so that the
    translation constraint reads ``C_i (T + b_e) = 0``.
    """
    x, e_phi, e_theta = _spherical_basis(phi, theta)
    v = -d_right_t[..., None] * e_phi - d_up_t[..., None] * e_theta
    N = x.shape[0]
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(N, 3, 3)
    Pi = I - x.unsqueeze(2) @ x.unsqueeze(1)
    C = _cross_matrix(v) @ Pi
    return x, v, C


def _normal_equations(terms, T_prov=None, foe_threshold=0.0):
    """Accumulate ``A = sum C_i^T C_i`` and ``g = -sum C_i^T C_i b_e`` over all
    columns of all eyes (the normal equations of ``min_T ||C T - d||^2``).

    When ``T_prov`` is given, columns inside each eye's focus-of-expansion cone
    (``||Pi_i t_hat_e||^2 <= foe_threshold``, with ``t_hat_e`` the direction of
    ``T_prov + b_e``) are dropped before accumulating.
    """
    A = g = None
    for (x, v, C, b) in terms:
        if T_prov is not None and foe_threshold > 0:
            t_hat = _safe_normalize(T_prov + b)
            proj = (x.unsqueeze(0) * t_hat.unsqueeze(1)).sum(-1)
            keep = (1.0 - proj ** 2) > foe_threshold
            C = C * keep[..., None, None]
        A_e = torch.einsum('tnij,tnik->tjk', C, C)
        g_e = -torch.einsum('tij,tj->ti', A_e, b)
        A = A_e if A is None else A + A_e
        g = g_e if g is None else g + g_e
    return A, g


def _safe_normalize(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _column_rho(x, v, T_eye):
    """Per-column ``rho_i = ||T_e||/Z_i`` from the de-rotated flow and ``T_e``.

    ``v_i = -rho_i Pi_i t_hat_e`` with ``t_hat_e = T_e / ||T_e||``, so
    ``rho_i = -(v_i . w_i)/||w_i||^2`` with ``w_i = Pi_i t_hat_e``.
    """
    t_hat = _safe_normalize(T_eye)
    proj = (x.unsqueeze(0) * t_hat.unsqueeze(1)).sum(-1)
    w = t_hat.unsqueeze(1) - proj.unsqueeze(-1) * x.unsqueeze(0)
    v_dot_w = (v * w).sum(-1)
    w_norm_sq = (w * w).sum(-1)
    return -v_dot_w / (w_norm_sq + 1e-12)


def _estimate_translation_two_eyes(derot, omega_star, foe_threshold=0.1):
    """Joint translation estimate over every column of every eye.

    Solves ``min_T ||C T - d||^2`` once over the stacked colinearity
    constraints (`_flow_constraint`), so the two eyes are combined in a single
    optimization rather than reduced to per-eye directions first. The
    rotation-induced parallax ``b_e = Omega x r_e`` makes ``d`` non-zero, fixing
    absolute scale. Two passes: an initial solve, then a re-solve after dropping
    focus-of-expansion columns. Timesteps whose parallax is too weak to
    condition the scale fall back to the single-sphere direction estimate
    (relative scale) over all columns.

    Eyes with no offset spread (zero/equal offsets) carry no parallax baseline
    at all, so the whole solve reduces to that single-sphere relative estimate.
    """
    device, dtype = omega_star.device, omega_star.dtype

    # Single-sphere (relative-scale) estimate over all columns of both eyes.
    # Always valid (monocular-equivalent) and used as the fallback.
    phi_all = torch.cat([eye.phi for eye, _, _ in derot])
    theta_all = torch.cat([eye.theta for eye, _, _ in derot])
    d_right_all = torch.cat([dr for _, dr, _ in derot], dim=1)
    d_up_all = torch.cat([du for _, _, du in derot], dim=1)
    T_rel, rho_rel = _estimate_translation_from_flow(
        phi_all, theta_all, d_right_all, d_up_all,
    )

    # No offset spread => no parallax baseline => absolute scale is impossible
    # for any rotation. Skip the inhomogeneous solve entirely (its RHS would be
    # ~0 and collapse T toward zero).
    offsets = torch.stack([eye.offset_vec(device, dtype) for eye, _, _ in derot])
    baseline = torch.cdist(offsets, offsets).max()
    if baseline < 1e-9:
        return T_rel, rho_rel

    terms = []
    for eye, d_right_t, d_up_t in derot:
        x, v, C = _flow_constraint(eye.phi, eye.theta, d_right_t, d_up_t)
        r = eye.offset_vec(device, dtype)
        b = torch.linalg.cross(omega_star, r.expand_as(omega_star))
        terms.append((x, v, C, b))

    # Pass 1 (all columns): provisional T and the parallax conditioning.
    A0, g0 = _normal_equations(terms)
    cond_ok = _scale_is_conditioned(A0)
    T_abs = _solve_batched(A0, g0)
    # Pass 2: drop focus-of-expansion columns, re-solve.
    A, g = _normal_equations(terms, T_abs, foe_threshold)
    T_abs = _solve_batched(A, g)

    T_star = torch.where(cond_ok[:, None], T_abs, T_rel)
    rho_star = torch.cat(
        [_column_rho(x, v, T_star + b) for (x, v, C, b) in terms], dim=1,
    )
    return T_star, rho_star


def _scale_is_conditioned(A):
    """True where the joint normal matrix has full-rank parallax (so absolute
    scale is recoverable), via the ratio of smallest to largest eigenvalue."""
    ev = torch.linalg.eigvalsh(A)
    return ev[:, 0] / ev[:, -1].clamp(min=1e-12) > _SCALE_COND_THRESHOLD


def _solve_batched(A, g):
    """Batched solve of ``A T = g`` (least-squares; tolerant of rank-deficient
    ``A`` in the degenerate timesteps, which are discarded by the caller)."""
    return torch.linalg.lstsq(A, g.unsqueeze(-1)).solution[..., 0]


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


def load_viewing_directions(data_folder: str | Path | None = None):
    """Load optic-lobe column phi/theta and stack both eyes.

    Reads every right-eye column of ``optic_lobe_columns.parquet`` in
    ``column_index`` order, the same column set and order as
    ``OpticLobeRegion`` and the generated stimuli.

    Args:
        data_folder: Path to the data root containing ``connectome_data/``.
            When ``None``, uses ``[data].data_folder`` of ``config.toml``.

    Returns:
        phi, theta: ``(2*N_columns,)`` float32 tensors: the right-eye columns,
        then their mirror across the midline (``-phi``) for the left eye.
    """
    if data_folder is None:
        from drosophila_model.config import get_data_folder, load_config
        data_folder = get_data_folder(load_config())
    data_folder = Path(data_folder)
    column_info_df = pd.read_parquet(
        data_folder / 'connectome_data' / 'baked' / 'optic_lobe_columns.parquet'
    ).sort_values('column_index')

    theta_half = torch.tensor(column_info_df['theta'].values, dtype=torch.float32)
    phi_half = torch.tensor(column_info_df['phi'].values, dtype=torch.float32)

    theta = torch.cat([theta_half, theta_half])
    phi = torch.cat([phi_half, -phi_half])
    return phi, theta


def generate_optic_flow(phi, theta, omega, T, Z, offset=None):
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
        offset: optional ``(3,)`` vector from the body center of mass to this
            eye. The eye's effective translation is ``T + omega x offset``
            (the rigid-body velocity of an off-center point). ``None`` / zero
            reproduces the single-center model.

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

    # Off-center eye: rigid-body velocity adds omega x offset to T.
    if offset is not None:
        offset = offset.to(device=omega.device, dtype=omega.dtype)
        T = T + torch.linalg.cross(omega, offset.expand_as(omega))

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
