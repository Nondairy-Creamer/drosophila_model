"""Tests for ego-motion estimation from synthetic optic flow.

Generates optic flow from known rotation/translation parameters, then checks
that ``OpticLobeRegion.estimate_ego_motion_from_flow`` recovers them.
"""
from pathlib import Path

import pytest
import torch

from drosophila_model.model_definition.ego_motion import (
    EyeFlow,
    estimate_from_flow,
    generate_optic_flow,
    load_viewing_directions,
)
from test_model_factory import (
    _resolve_data_folder,
    build_model_from_test_config,
    load_test_config,
)

TRANSLATION_THRESHOLD = 1e-5


def _align_translation_sign(T_pred, T_true):
    """Flip T_pred to the sign-consistent hemisphere (direction is only up to sign)."""
    T_dir = T_pred / T_pred.norm() if T_pred.norm() > 0 else T_pred
    if (T_dir - T_true).norm() > (T_dir + T_true).norm():
        T_dir = -T_dir
    return T_dir


@pytest.fixture(scope="module")
def viewing_dirs():
    """Load viewing directions once for the whole module."""
    config, repo_root = load_test_config()
    data_folder = _resolve_data_folder(repo_root, config["data"]["data_folder"])
    phi, theta = load_viewing_directions(data_folder)
    return phi, theta


@pytest.fixture(scope="module")
def model():
    """Build a model instance for calling ego-motion estimation."""
    m, _, _, _ = build_model_from_test_config(readouts=[])
    m.eval()
    return m


def test_pure_rotation(viewing_dirs, model):
    """Recover rotation when translation is zero."""
    phi, theta = viewing_dirs
    omega_true = torch.tensor([0.1, -0.2, 0.15])
    T_true = torch.tensor([0.0, 0.0, 0.0])

    d_right, d_up = generate_optic_flow(phi, theta, omega_true, T_true, 1.0)
    omega_pred, T_pred, _ = model.regions["optic_lobe"].estimate_ego_motion_from_flow(
        phi, theta, d_right, d_up,
    )

    assert (omega_pred - omega_true).norm() < 1e-4
    assert T_pred.norm() < TRANSLATION_THRESHOLD


def test_pure_translation(viewing_dirs, model):
    """Recover translation direction when rotation is zero."""
    phi, theta = viewing_dirs
    N = len(theta)
    omega_true = torch.tensor([0.0, 0.0, 0.0])
    T_true = torch.tensor([1.0, 0.5, -0.3])
    T_true = T_true / T_true.norm()
    Z = torch.ones(N) * 2.0

    d_right, d_up = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    omega_pred, T_pred, _ = model.regions["optic_lobe"].estimate_ego_motion_from_flow(
        phi, theta, d_right, d_up,
    )

    T_pred_dir = _align_translation_sign(T_pred, T_true)
    # Partial sphere coverage causes some rotation-translation coupling
    assert omega_pred.norm() < 0.25
    assert (T_pred_dir - T_true).norm() < 0.25


def test_combined_motion(viewing_dirs, model):
    """Recover both rotation and translation simultaneously."""
    phi, theta = viewing_dirs
    N = len(theta)
    omega_true = torch.tensor([0.05, -0.1, 0.08])
    T_true = torch.tensor([0.7, -0.4, 0.6])
    T_true = T_true / T_true.norm()
    Z = torch.ones(N) * 3.0

    d_right, d_up = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    omega_pred, T_pred, _ = model.regions["optic_lobe"].estimate_ego_motion_from_flow(
        phi, theta, d_right, d_up,
    )

    T_pred_dir = _align_translation_sign(T_pred, T_true)
    # Tolerances reflect the geometry produced by format_connectome_data.py
    # (phi span ~2.5 rad). Narrower spans give less leverage on the
    # cross-eye rotation axis.
    assert (omega_pred - omega_true).norm() < 0.15
    assert (T_pred_dir - T_true).norm() < 0.15


def test_varying_depth(viewing_dirs, model):
    """Recover motion parameters with spatially varying depth."""
    phi, theta = viewing_dirs
    omega_true = torch.tensor([0.02, 0.03, -0.05])
    T_true = torch.tensor([1.0, 0.0, 0.0])

    x_component = torch.sin(theta) * torch.cos(phi)
    Z = 5.0 - 2.0 * x_component

    d_right, d_up = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    omega_pred, T_pred, rho_pred = model.regions["optic_lobe"].estimate_ego_motion_from_flow(
        phi, theta, d_right, d_up,
    )

    T_pred_dir = _align_translation_sign(T_pred, T_true)

    assert (omega_pred - omega_true).norm() < 0.1
    assert (T_pred_dir - T_true).norm() < 0.1

    # rho_i = ||T||/Z_i is the recoverable per-point quantity; should track 1/Z.
    inv_z = 1.0 / Z
    depth_corr = torch.corrcoef(torch.stack([inv_z, rho_pred]))[0, 1]
    assert depth_corr > 0.5


def test_noisy_flow(viewing_dirs, model):
    """Estimation degrades gracefully with additive noise."""
    phi, theta = viewing_dirs
    N = len(theta)
    omega_true = torch.tensor([0.1, -0.05, 0.12])
    T_true = torch.tensor([0.5, 0.5, 0.7071])
    T_true = T_true / T_true.norm()
    Z = torch.ones(N) * 4.0

    d_right, d_up = generate_optic_flow(phi, theta, omega_true, T_true, Z)

    torch.manual_seed(42)
    noise_level = 0.01
    d_right_noisy = d_right + noise_level * torch.randn_like(d_right)
    d_up_noisy = d_up + noise_level * torch.randn_like(d_up)

    omega_pred, T_pred, _ = model.regions["optic_lobe"].estimate_ego_motion_from_flow(
        phi, theta, d_right_noisy, d_up_noisy,
    )

    T_pred_dir = _align_translation_sign(T_pred, T_true)
    assert (omega_pred - omega_true).norm() < 0.15
    assert (T_pred_dir - T_true).norm() < 0.15


def _fibonacci_sphere(n=800):
    """Dense, near-uniform sphere coverage (isolates the solver math from the
    partial-sphere coverage of real fly columns)."""
    i = torch.arange(n, dtype=torch.float32)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = torch.acos(z.clamp(-1.0, 1.0))
    phi = torch.remainder(2.0 * torch.pi * i / golden, 2.0 * torch.pi)
    return phi, theta


def test_two_eye_absolute_scale():
    """Two offset eyes recover absolute translation magnitude, not just
    direction, from the rotation-induced inter-eye parallax."""
    phi, theta = _fibonacci_sphere(800)
    Z = torch.full((phi.numel(),), 2.0)

    omega_true = torch.tensor([0.3, -0.2, 0.4])
    T_true = torch.tensor([0.5, -0.3, 0.2])  # absolute, NOT unit-normalised
    r_right = torch.tensor([0.3, -0.25, 0.2])
    r_left = torch.tensor([0.3, 0.25, 0.2])  # sagittal mirror (+y is left)

    dr_r, du_r = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_right)
    dr_l, du_l = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_left)

    eyes = [
        EyeFlow(phi=phi, theta=theta, d_right=dr_r, d_up=du_r, offset=r_right),
        EyeFlow(phi=phi, theta=theta, d_right=dr_l, d_up=du_l, offset=r_left),
    ]
    omega_pred, T_pred, _ = estimate_from_flow(eyes)

    assert (omega_pred - omega_true).norm() < 1e-3
    # Full vector (including magnitude) recovered, not just direction.
    assert (T_pred - T_true).norm() < 1e-2


def test_two_eye_degenerate_scale_falls_back_to_direction():
    """With zero rotation there is no parallax, so absolute scale is
    unrecoverable; the solver should still return the right direction."""
    phi, theta = _fibonacci_sphere(800)
    Z = torch.full((phi.numel(),), 2.0)

    omega_true = torch.tensor([0.0, 0.0, 0.0])
    T_true = torch.tensor([0.5, -0.3, 0.2])
    r_right = torch.tensor([0.3, -0.25, 0.2])
    r_left = torch.tensor([0.3, 0.25, 0.2])

    dr_r, du_r = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_right)
    dr_l, du_l = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_left)

    eyes = [
        EyeFlow(phi=phi, theta=theta, d_right=dr_r, d_up=du_r, offset=r_right),
        EyeFlow(phi=phi, theta=theta, d_right=dr_l, d_up=du_l, offset=r_left),
    ]
    _, T_pred, _ = estimate_from_flow(eyes)

    T_pred_dir = _align_translation_sign(T_pred, T_true / T_true.norm())
    assert (T_pred_dir - T_true / T_true.norm()).norm() < 0.05


def test_two_eye_absolute_scale_noisy():
    """Joint two-eye solve degrades gracefully: absolute scale stays close
    under additive flow noise."""
    phi, theta = _fibonacci_sphere(1200)
    Z = torch.full((phi.numel(),), 2.0)

    omega_true = torch.tensor([0.3, -0.2, 0.4])
    T_true = torch.tensor([0.5, -0.3, 0.2])
    r_right = torch.tensor([0.3, -0.25, 0.2])
    r_left = torch.tensor([0.3, 0.25, 0.2])

    dr_r, du_r = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_right)
    dr_l, du_l = generate_optic_flow(phi, theta, omega_true, T_true, Z, offset=r_left)

    torch.manual_seed(0)
    noise = 0.01
    dr_r = dr_r + noise * torch.randn_like(dr_r)
    du_r = du_r + noise * torch.randn_like(du_r)
    dr_l = dr_l + noise * torch.randn_like(dr_l)
    du_l = du_l + noise * torch.randn_like(du_l)

    eyes = [
        EyeFlow(phi=phi, theta=theta, d_right=dr_r, d_up=du_r, offset=r_right),
        EyeFlow(phi=phi, theta=theta, d_right=dr_l, d_up=du_l, offset=r_left),
    ]
    omega_pred, T_pred, _ = estimate_from_flow(eyes)

    assert (omega_pred - omega_true).norm() < 0.02
    # Absolute magnitude within ~25% under this noise level.
    assert (T_pred - T_true).norm() / T_true.norm() < 0.25
