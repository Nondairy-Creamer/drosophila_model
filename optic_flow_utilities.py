import torch

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


def get_self_motion(theta, phi, d_theta, d_phi):
    """
    Estimate self-motion (rotation and translation) from optic flow.
    
    Args:
        theta: (N,) polar angles of viewing directions
        phi: (N,) azimuthal angles of viewing directions  
        d_theta: (N,) angular flow in theta direction. up / down is positive / negative respectively
        d_phi: (N,) angular flow in phi direction. right / left is positive / negative respectively
    
    Returns:
        omega_star: (3,) estimated rotation vector
        T_star: (3,) estimated translation direction (unit vector)
        Z_star: (N,) estimated depths
        translation_magnitude: scalar indicating strength of translation signal
            (RMS of de-rotated flow; near zero means no translation)
    """

    N = theta.shape[0]

    # === Convert to 3D viewing directions (N,3) ===
    x = torch.stack((
        torch.sin(theta)*torch.cos(phi),
        torch.sin(theta)*torch.sin(phi),
        torch.cos(theta)
    ), dim=1)                              # (N,3)

    # === Convert angular flow to 3D tangent vectors u_i ∈ R^3 ===
    # Basis vectors of sphere
    e_theta = torch.stack((
        torch.cos(theta)*torch.cos(phi),
        torch.cos(theta)*torch.sin(phi),
        -torch.sin(theta)
    ), dim=1)                              # (N,3)

    e_phi = torch.stack((
        -torch.sin(phi),
         torch.cos(phi),
         torch.zeros_like(phi)
    ), dim=1)                              # (N,3)

    # Negate d_theta because e_theta points downward but we want up=positive
    u = -d_theta[:, None] * e_theta + d_phi[:, None] * e_phi   # (N,3)

    # === Rotation estimation: solve u ≈ [x]_× Ω ===
    A = cross_matrix(x)                    # (N,3,3)
    A_flat = A.reshape(3*N, 3)
    u_flat = u.reshape(3*N)

    omega_star = torch.linalg.lstsq(A_flat, u_flat)[0]

    # === De-rotated flow: v_i = u_i - [x_i]_× Ω* ===
    v = u - (A @ omega_star)               # (N,3)
    
    # === Measure translation signal strength ===
    # RMS of de-rotated flow - if near zero, there's no translation
    translation_magnitude = torch.sqrt((v**2).mean())

    # === Projection matrices Π_i = I - x_i x_iᵀ ===
    # Each Pi is (N,3,3)
    outer = x.unsqueeze(2) @ x.unsqueeze(1)   # (N,3,3)
    I = torch.eye(3, device=x.device).expand(N,3,3)
    P = I - outer

    # === Construct stacked C matrix: each block = [v_i]_× Π_i ===
    vx = cross_matrix(v)                   # (N,3,3)
    C_blocks = vx @ P                      # (N,3,3)

    C = C_blocks.reshape(3*N, 3)           # (3N,3)

    # === Solve C T = 0 via smallest singular vector ===
    _, _, Vh = torch.linalg.svd(C, full_matrices=False)
    # scale T estimate by translation magnitude so that it is proportional
    # to true translation across frames
    T_star = Vh[-1] * translation_magnitude    # smallest singular value direction

    # === w_i = Π_i T* ===
    w_i = (P @ T_star)                     # (N,3)

    # Depth from: v = -(1/Z) * Π * T  =>  Z = -||w||² / (v · w)
    Z_star = -(w_i.norm(dim=1)**2) / (v * w_i).sum(dim=1)

    return omega_star, T_star, Z_star
