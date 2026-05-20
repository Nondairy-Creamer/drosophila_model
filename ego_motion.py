"""Ego-motion math: SO(3) rotations and velocity integration.

Pure-tensor helpers with no dependency on the rest of the codebase. Used by
:class:`neural_models.BaseRnn` (which wraps the flow -> ego-motion solver as
an instance method) and by stimulus / analysis scripts that need to integrate
target trajectories.
"""

from __future__ import annotations

import numpy as np
import torch


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
