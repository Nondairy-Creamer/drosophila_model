"""Tests for ego-motion estimation from synthetic optic flow.

Generates optic flow from known rotation/translation parameters, then checks
that ``OpticLobeRegion.estimate_ego_motion_from_flow`` recovers them.
"""
from pathlib import Path

import pytest
import torch

from drosophila_model.analysis.analysis_utilities import generate_optic_flow, load_viewing_directions
from test_model_factory import build_model_from_test_config, load_test_config

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
    connectome_folder = Path(config["data"]["connectome_folder"])
    if not connectome_folder.is_absolute():
        connectome_folder = (repo_root / connectome_folder).resolve()
    phi, theta = load_viewing_directions(connectome_folder)
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
