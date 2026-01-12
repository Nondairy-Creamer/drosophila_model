"""
Test script for optic flow self-motion estimation.
Generates synthetic optic flow from known rotations and translations,
then verifies the model can recover these parameters.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_models import NeuralRnn

def cross_matrix(a):
    """
    a: (..., 3) tensor
    returns (..., 3, 3) skew-symmetric matrix [a]_x
    """
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    O = torch.zeros_like(ax)
    return torch.stack([
        torch.stack([   O, -az,  ay], dim=-1),
        torch.stack([  az,   O, -ax], dim=-1),
        torch.stack([ -ay,  ax,   O], dim=-1)
    ], dim=-2)

# Load column info with theta/phi viewing directions
connectome_folder = Path('../connectome_data')
column_info_df = pd.read_parquet(connectome_folder / 'optic_lobes_column_info.parquet')

# Extract theta and phi as tensors
# Include both hemispheres: original phi and negated phi (bilateral visual field)
theta_half = torch.tensor(column_info_df['theta'].values, dtype=torch.float32)
phi_half = torch.tensor(column_info_df['phi'].values, dtype=torch.float32)

theta = torch.cat([theta_half, theta_half])
phi = torch.cat([phi_half, -phi_half])
N = len(theta)

print(f"Loaded {N} viewing directions")
print(f"Theta range: [{theta.min():.3f}, {theta.max():.3f}] rad")
print(f"Phi range: [{phi.min():.3f}, {phi.max():.3f}] rad")


def generate_optic_flow(phi, theta, omega, T, Z):
    """
    Generate synthetic optic flow given rotation omega, translation T, and depths Z.
    
    Args:
        phi: (N,) azimuthal angles
        theta: (N,) polar angles
        omega: (3,) rotation vector
        T: (3,) translation vector
        Z: (N,) or scalar depth values
    
    Returns:
        d_phi: (N,) angular flow in phi direction
        d_theta: (N,) angular flow in theta direction
    """
    N = theta.shape[0]
    
    # Convert to 3D viewing directions (N,3)
    x = torch.stack((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ), dim=1)
    
    # Basis vectors on the sphere
    e_theta = torch.stack((
        torch.cos(theta) * torch.cos(phi),
        torch.cos(theta) * torch.sin(phi),
        -torch.sin(theta)
    ), dim=1)
    
    e_phi = torch.stack((
        -torch.sin(phi),
        torch.cos(phi),
        torch.zeros_like(phi)
    ), dim=1)
    
    # Rotation contribution: u = Ω × x = -x × Ω = -[x]_× Ω
    # Physical: leftward rotation (positive omega_z) → rightward image motion (positive d_phi)
    A = cross_matrix(x)  # (N, 3, 3)
    u_rot = -(A @ omega)  # (N, 3) - negate for correct physical sign
    
    # Translation contribution: u_trans = (1/Z) * Pi * T
    # where Pi = I - x x^T (projection onto tangent plane)
    outer = x.unsqueeze(2) @ x.unsqueeze(1)  # (N, 3, 3)
    I = torch.eye(3).expand(N, 3, 3)
    Pi = I - outer
    
    # Translation flow is OPPOSITE to motion direction (objects move backward when you move forward)
    if torch.is_tensor(Z) and Z.dim() > 0:
        u_trans = -(Pi @ T) / Z.unsqueeze(1)  # (N, 3)
    else:
        u_trans = -(Pi @ T) / Z  # (N, 3)
    
    # Total flow in 3D
    u = u_rot + u_trans
    
    # Project onto spherical basis to get angular flow
    # d_phi: positive = rightward motion on retina (leftward observer rotation)
    # d_theta: positive = upward motion on retina (downward observer motion)
    # Negate d_theta because e_theta points downward but we want up=positive
    d_theta = -(u * e_theta).sum(dim=1)
    d_phi = (u * e_phi).sum(dim=1)
    
    return d_phi, d_theta


def format_vec(v, name):
    """Format a 3D vector for display."""
    if torch.is_tensor(v):
        v = v.numpy()
    return f"{name} = [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"


# Threshold for detecting zero translation (based on ||T_star||)
TRANSLATION_THRESHOLD = 1e-5


def plot_flow_field(theta, phi, d_theta, d_phi, title, 
                    omega_true, T_true, omega_pred, T_pred, 
                    filename, extra_info=None):
    """
    Plot the optic flow field on the flattened angular view.
    
    Args:
        theta, phi: viewing directions
        d_theta, d_phi: flow components
        title: plot title
        omega_true, T_true: ground truth rotation and translation
        omega_pred, T_pred: predicted rotation and translation
        filename: output filename
        extra_info: optional extra text to display
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Convert to numpy
    phi_np = phi.numpy()
    theta_np = theta.numpy()
    d_phi_np = d_phi.numpy()
    d_theta_np = d_theta.numpy()
    
    # Compute flow magnitude for coloring
    flow_mag = np.sqrt(d_phi_np**2 + d_theta_np**2)
    
    # Subsample for cleaner quiver plot
    step = max(1, len(phi_np) // 300)
    idx = np.arange(0, len(phi_np), step)
    
    # Scale arrows for visibility - normalize by max magnitude
    max_mag = flow_mag.max()
    if max_mag > 0:
        scale_factor = 0.15  # Target arrow length relative to plot
        scale = max_mag / scale_factor
    else:
        scale = 1
    
    # Plot quiver (flow field)
    q = ax.quiver(phi_np[idx], theta_np[idx], 
                  d_phi_np[idx], d_theta_np[idx],
                  flow_mag[idx], cmap='plasma',
                  angles='xy', scale_units='xy', scale=scale,
                  width=0.008, headwidth=3, headlength=4, headaxislength=3)
    
    # Colorbar
    cbar = plt.colorbar(q, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Flow Magnitude', fontsize=11)
    
    # Axis labels and styling
    ax.set_xlabel('Phi (rad)', fontsize=12)
    ax.set_ylabel('Theta (rad)', fontsize=12)
    ax.set_xlim(np.pi, -np.pi)  # Flip X so negative phi is on right
    ax.set_ylim(np.pi, 0)       # Flip Y so theta=0 is at top
    ax.axhline(y=np.pi/2, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Compute errors
    omega_err = (omega_pred - omega_true).norm().item()
    T_err = (T_pred - T_true).norm().item()
    
    # Text box with ground truth and predictions
    text_lines = [
        "─── Ground Truth ───",
        f"ω = [{omega_true[0]:+.3f}, {omega_true[1]:+.3f}, {omega_true[2]:+.3f}]",
        f"T = [{T_true[0]:+.3f}, {T_true[1]:+.3f}, {T_true[2]:+.3f}]",
        "",
        "─── Predicted ───",
        f"ω = [{omega_pred[0]:+.3f}, {omega_pred[1]:+.3f}, {omega_pred[2]:+.3f}]",
        f"T = [{T_pred[0]:+.3f}, {T_pred[1]:+.3f}, {T_pred[2]:+.3f}]",
        "",
        "─── Errors ───",
        f"ω error: {omega_err:.6f}",
        f"T error: {T_err:.6f}",
    ]
    
    if extra_info:
        text_lines.extend(["", extra_info])
    
    text = "\n".join(text_lines)
    
    # Add text box to the right of the plot
    props = dict(boxstyle='round,pad=0.6', facecolor='#f8f8f8', alpha=0.95, edgecolor='#cccccc')
    fig.text(0.98, 0.5, text, fontsize=9, verticalalignment='center', 
             horizontalalignment='left', fontfamily='monospace', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.68)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def test_rotation_only():
    """Test recovery of pure rotation (no translation)."""
    print("\n" + "="*60)
    print("TEST 1: Pure Rotation")
    print("="*60)
    
    # Ground truth rotation
    omega_true = torch.tensor([0.1, -0.2, 0.15])
    T_true = torch.tensor([0.0, 0.0, 0.0])
    Z = 1.0  # Depth doesn't matter for pure rotation
    
    print(f"True rotation omega: {omega_true.numpy()}")
    
    # Generate flow
    d_phi, d_theta = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    
    # Recover self-motion
    omega_pred, T_pred, Z_pred = NeuralRnn.estimate_self_motion_from_flow(phi, theta, d_phi, d_theta)
    
    # ||T_pred|| now represents translation magnitude (scaled by 1/Z)
    trans_mag = T_pred.norm().item()
    
    # Detect zero translation
    has_translation = trans_mag > TRANSLATION_THRESHOLD
    if not has_translation:
        T_pred = torch.zeros(3)  # Set T to zero if no translation detected
    
    print(f"Predicted omega:     {omega_pred.numpy()}")
    print(f"Rotation error:      {(omega_pred - omega_true).norm():.6f}")
    print(f"Translation signal:  {trans_mag:.6f} {'(detected)' if has_translation else '(none - T set to 0)'}")
    
    # Plot flow field
    plot_flow_field(
        theta, phi, d_theta, d_phi,
        title="Pure Rotation Flow Field",
        omega_true=omega_true, T_true=T_true,
        omega_pred=omega_pred, T_pred=T_pred,
        filename='figures/analysis/flow_rotation_only.png',
        extra_info=f"Trans signal: {trans_mag:.2e}\n(T=0, rotation only)"
    )
    
    return omega_pred, T_pred


def test_translation_only():
    """Test recovery of pure translation (no rotation)."""
    print("\n" + "="*60)
    print("TEST 2: Pure Translation")
    print("="*60)
    
    # Ground truth translation (normalized direction)
    omega_true = torch.tensor([0.0, 0.0, 0.0])
    T_true = torch.tensor([1.0, 0.5, -0.3])
    T_true = T_true / T_true.norm()  # Normalize
    Z = torch.ones(N) * 2.0  # Uniform depth
    
    print(f"True translation T:  {T_true.numpy()}")
    
    # Generate flow
    d_phi, d_theta = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    
    # Recover self-motion
    omega_pred, T_pred, Z_pred = NeuralRnn.estimate_self_motion_from_flow(phi, theta, d_phi, d_theta)
    trans_mag = T_pred.norm().item()
    
    # Normalize T_pred to compare direction (T_pred is now scaled)
    T_pred_dir = T_pred / T_pred.norm() if T_pred.norm() > 0 else T_pred
    
    # Translation direction is only determined up to sign
    if (T_pred_dir - T_true).norm() > (T_pred_dir + T_true).norm():
        T_pred = -T_pred
        T_pred_dir = -T_pred_dir
        Z_pred = -Z_pred
    
    print(f"Predicted T (dir):   {T_pred_dir.numpy()}")
    print(f"Predicted ||T||:     {trans_mag:.6f}")
    print(f"Direction error:     {(T_pred_dir - T_true).norm():.6f}")
    print(f"True depth:          {Z[0]:.3f}")
    print(f"Predicted depth (mean): {Z_pred.mean():.3f}")
    
    # Plot flow field
    plot_flow_field(
        theta, phi, d_theta, d_phi,
        title="Pure Translation Flow Field",
        omega_true=omega_true, T_true=T_true,
        omega_pred=omega_pred, T_pred=T_pred_dir,
        filename='figures/analysis/flow_translation_only.png',
        extra_info=f"||T|| = {trans_mag:.3f}\n(ω=0, depth Z={Z[0]:.1f})"
    )
    
    return omega_pred, T_pred, Z_pred


def test_combined_motion():
    """Test recovery of combined rotation and translation."""
    print("\n" + "="*60)
    print("TEST 3: Combined Rotation + Translation")
    print("="*60)
    
    # Ground truth
    omega_true = torch.tensor([0.05, -0.1, 0.08])
    T_true = torch.tensor([0.7, -0.4, 0.6])
    T_true = T_true / T_true.norm()
    Z = torch.ones(N) * 3.0
    
    print(f"True omega:          {omega_true.numpy()}")
    print(f"True T:              {T_true.numpy()}")
    
    # Generate flow
    d_phi, d_theta = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    
    # Recover self-motion
    omega_pred, T_pred, Z_pred = NeuralRnn.estimate_self_motion_from_flow(phi, theta, d_phi, d_theta)
    trans_mag = T_pred.norm().item()
    
    # Normalize T_pred to compare direction
    T_pred_dir = T_pred / T_pred.norm() if T_pred.norm() > 0 else T_pred
    
    # Translation direction is only determined up to sign
    if (T_pred_dir - T_true).norm() > (T_pred_dir + T_true).norm():
        T_pred = -T_pred
        T_pred_dir = -T_pred_dir
        Z_pred = -Z_pred
    
    print(f"Predicted omega:     {omega_pred.numpy()}")
    print(f"Predicted T (dir):   {T_pred_dir.numpy()}")
    print(f"Predicted ||T||:     {trans_mag:.6f}")
    print(f"Rotation error:      {(omega_pred - omega_true).norm():.6f}")
    print(f"Direction error:     {(T_pred_dir - T_true).norm():.6f}")
    
    # Plot flow field
    plot_flow_field(
        theta, phi, d_theta, d_phi,
        title="Combined Rotation + Translation Flow Field",
        omega_true=omega_true, T_true=T_true,
        omega_pred=omega_pred, T_pred=T_pred_dir,
        filename='figures/analysis/flow_combined.png',
        extra_info=f"||T|| = {trans_mag:.3f}\n(depth Z={Z[0]:.1f})"
    )
    
    return omega_pred, T_pred, Z_pred


def test_varying_depth():
    """Test with varying depth across the visual field."""
    print("\n" + "="*60)
    print("TEST 4: Varying Depth")
    print("="*60)
    
    # Ground truth
    # Coordinate convention: +X = Forward, +Y = Right, +Z = Up
    omega_true = torch.tensor([0.02, 0.03, -0.05])
    T_true = torch.tensor([1.0, 0.0, 0.0])  # Forward motion (+X)
    
    # Depth varies with viewing direction (simulate approaching a tilted plane)
    # Points in the forward direction (+X) are closer
    x_component = torch.sin(theta) * torch.cos(phi)
    Z = 5.0 - 2.0 * x_component  # Closer in front, farther behind
    
    print(f"True omega:          {omega_true.numpy()}")
    print(f"True T:              {T_true.numpy()}")
    print(f"Depth range:         [{Z.min():.2f}, {Z.max():.2f}]")
    
    # Generate flow
    d_phi, d_theta = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    
    # Recover self-motion
    omega_pred, T_pred, Z_pred = NeuralRnn.estimate_self_motion_from_flow(phi, theta, d_phi, d_theta)
    trans_mag = T_pred.norm().item()
    
    # Normalize T_pred to compare direction
    T_pred_dir = T_pred / T_pred.norm() if T_pred.norm() > 0 else T_pred
    
    # Translation direction is only determined up to sign
    if (T_pred_dir - T_true).norm() > (T_pred_dir + T_true).norm():
        T_pred = -T_pred
        T_pred_dir = -T_pred_dir
        Z_pred = -Z_pred
    
    print(f"Predicted omega:     {omega_pred.numpy()}")
    print(f"Predicted T (dir):   {T_pred_dir.numpy()}")
    print(f"Predicted ||T||:     {trans_mag:.6f}")
    print(f"Rotation error:      {(omega_pred - omega_true).norm():.6f}")
    print(f"Direction error:     {(T_pred_dir - T_true).norm():.6f}")
    print(f"Pred depth range:    [{Z_pred.min():.2f}, {Z_pred.max():.2f}]")
    
    # Correlation between true and predicted depth
    corr = torch.corrcoef(torch.stack([Z, Z_pred]))[0, 1]
    print(f"Depth correlation:   {corr:.4f}")
    
    # Plot flow field
    plot_flow_field(
        theta, phi, d_theta, d_phi,
        title="Varying Depth Flow Field (Forward Motion)",
        omega_true=omega_true, T_true=T_true,
        omega_pred=omega_pred, T_pred=T_pred_dir,
        filename='figures/analysis/flow_varying_depth.png',
        extra_info=f"||T|| = {trans_mag:.3f}\nZ ∈ [{Z.min():.1f}, {Z.max():.1f}]\nDepth corr: {corr:.4f}"
    )
    
    return omega_pred, T_pred, Z_pred


def test_noisy_flow():
    """Test robustness to noisy measurements."""
    print("\n" + "="*60)
    print("TEST 5: Noisy Flow")
    print("="*60)
    
    # Ground truth
    omega_true = torch.tensor([0.1, -0.05, 0.12])
    T_true = torch.tensor([0.5, 0.5, 0.7071])
    T_true = T_true / T_true.norm()
    Z = torch.ones(N) * 4.0
    
    print(f"True omega:          {omega_true.numpy()}")
    print(f"True T:              {T_true.numpy()}")
    
    # Generate clean flow
    d_phi, d_theta = generate_optic_flow(phi, theta, omega_true, T_true, Z)
    
    # Add noise
    noise_level = 0.01
    torch.manual_seed(42)
    d_theta_noisy = d_theta + noise_level * torch.randn_like(d_theta)
    d_phi_noisy = d_phi + noise_level * torch.randn_like(d_phi)
    
    print(f"Noise level:         {noise_level}")
    print(f"Flow magnitude:      {torch.sqrt(d_theta**2 + d_phi**2).mean():.4f}")
    
    # Recover self-motion
    omega_pred, T_pred, Z_pred = NeuralRnn.estimate_self_motion_from_flow(phi, theta, d_phi_noisy, d_theta_noisy)
    trans_mag = T_pred.norm().item()
    
    # Normalize T_pred to compare direction
    T_pred_dir = T_pred / T_pred.norm() if T_pred.norm() > 0 else T_pred
    
    # Translation direction is only determined up to sign
    if (T_pred_dir - T_true).norm() > (T_pred_dir + T_true).norm():
        T_pred = -T_pred
        T_pred_dir = -T_pred_dir
        Z_pred = -Z_pred
    
    print(f"Predicted omega:     {omega_pred.numpy()}")
    print(f"Predicted T (dir):   {T_pred_dir.numpy()}")
    print(f"Predicted ||T||:     {trans_mag:.6f}")
    print(f"Rotation error:      {(omega_pred - omega_true).norm():.6f}")
    print(f"Direction error:     {(T_pred_dir - T_true).norm():.6f}")
    
    return omega_pred, T_pred, Z_pred


def visualize_coverage():
    """Visualize the visual field coverage."""
    print("\n" + "="*60)
    print("Visualizing Visual Field Coverage")
    print("="*60)
    
    # Convert to Cartesian for visualization
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x.numpy(), y.numpy(), z.numpy(), c=theta.numpy(), cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Visual Field ({N} points)')
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(xs, ys, zs, color='gray', alpha=0.1)
    
    # 2D projection (theta vs phi)
    ax2 = fig.add_subplot(122)
    ax2.scatter(phi.numpy(), theta.numpy(), c=theta.numpy(), cmap='viridis', s=10, alpha=0.7)
    ax2.set_xlabel('Phi (rad)')
    ax2.set_ylabel('Theta (rad)')
    ax2.set_title(f'Angular Coverage ({N} points)')
    ax2.axhline(y=np.pi/2, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlim(np.pi, -np.pi)  # Flip X so negative phi is on right
    ax2.set_ylim(np.pi, 0)       # Flip Y so theta=0 is at top
    
    plt.tight_layout()
    plt.savefig('figures/analysis/visual_field_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure to figures/analysis/visual_field_coverage.png")


if __name__ == "__main__":
    print("Optic Flow Self-Motion Estimation Tests")
    print("Using Drosophila connectome visual field data")
    
    test_rotation_only()
    test_translation_only()
    test_combined_motion()
    test_varying_depth()
    test_noisy_flow()
    
    # Visualize coverage
    visualize_coverage()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
