import cv2
import numpy as np

SQRT_3 = np.sqrt(3)
HALF_SQRT_3 = SQRT_3/2

def flip_horizontal(p, q):
    return -p - q, q

def rotate_60(p, q):
    return -q, p - q

def rotate_minus_60(p, q):
    return p + q, -p

def rays_from_pixels(image, K, xi, D=None, correct_distortion=False):
    # return rays from pixels in image given omnidirectional camera model parameters
    # Parameters:
    # image: image to get rays from
    # K: camera matrix
    # xi: skew parameter
    # D: distortion coefficients
    # correct_distortion: whether to correct distortion
    # Returns:
    # rays: rays from pixels in image
    H,W = image.shape[:2]
    
    # unpack K
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # Pixel grid - store original pixel coordinates
    v, u = np.indices((H, W))  # v is row (y), u is column (x)
    
    xi = float(xi)
    
    # Convert to normalized coordinates relative to optical center
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x*x + y*y
    
    # optional undistortion (small effect if D small)
    if correct_distortion and D is not None:
        k1, k2, p1, p2 = D.ravel()
        for _ in range(5):  # iterative inverse distortion
            r2 = x*x + y*y
            delta_x = 2*p1*x*y + p2*(r2 + 2*x*x)
            delta_y = p1*(r2 + 2*y*y) + 2*p2*x*y
            x = ( (u - cx)/fx - delta_x ) / (1 + k1*r2 + k2*r2*r2)
            y = ( (v - cy)/fy - delta_y ) / (1 + k1*r2 + k2*r2*r2)
            r2 = x*x + y*y
    z = (1 - xi*r2) / (xi + np.sqrt(1 + (1 - xi**2)*r2))
    vec = np.stack([x,y,z], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec  # shape (..., 3)


def ray_from_pixel(u, v, K, xi, D=None, correct_distortion=False):
    """
    Return ray(s) from pixel coordinate(s) given omnidirectional camera model parameters.
    Supports fractional pixel coordinates.
    
    Parameters:
    -----------
    u : float or array-like
        Pixel column coordinate(s) (x-coordinate). Can be fractional.
    v : float or array-like
        Pixel row coordinate(s) (y-coordinate). Can be fractional.
    K : ndarray (3, 3)
        Camera matrix
    xi : float
        Skew parameter
    D : ndarray, optional
        Distortion coefficients (4 elements: k1, k2, p1, p2)
    correct_distortion : bool
        Whether to correct distortion
        
    Returns:
    --------
    rays : ndarray
        Ray direction vector(s). Shape is (3,) for single point, (N, 3) for N points.
    """
    # Convert to numpy arrays and ensure they're the same shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Handle scalar inputs
    if u.ndim == 0:
        u = u[None]
        v = v[None]
        scalar_input = True
    else:
        scalar_input = False
    
    # Ensure u and v have compatible shapes
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    
    # Unpack K
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    xi = float(xi)
    
    # Convert to normalized coordinates relative to optical center
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x*x + y*y
    
    # Optional undistortion (small effect if D small)
    if correct_distortion and D is not None:
        k1, k2, p1, p2 = D.ravel()
        for _ in range(5):  # iterative inverse distortion
            r2 = x*x + y*y
            delta_x = 2*p1*x*y + p2*(r2 + 2*x*x)
            delta_y = p1*(r2 + 2*y*y) + 2*p2*x*y
            x = ( (u - cx)/fx - delta_x ) / (1 + k1*r2 + k2*r2*r2)
            y = ( (v - cy)/fy - delta_y ) / (1 + k1*r2 + k2*r2*r2)
            r2 = x*x + y*y
    
    z = (1 - xi*r2) / (xi + np.sqrt(1 + (1 - xi**2)*r2))
    vec = np.stack([x, y, z], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    
    # Return scalar if input was scalar
    if scalar_input:
        return vec[0]
    
    return vec  # shape (N, 3) for N points


def rotate_coords(x, y, angle_degrees):
    """
    Rotates x and y coordinates by a specified angle (in degrees) around the origin.
    
    Args:
        x (float or array-like): The x-coordinate(s).
        y (float or array-like): The y-coordinate(s).
        angle_degrees (float): The rotation angle in degrees.
        
    Returns:
        tuple: (new_x, new_y) the rotated coordinates.
    """
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    # Define the 2D rotation matrix for a counter-clockwise rotation
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    # Combine x and y into a single numpy array of coordinates (e.g., as column vectors)
    # The shape will be (2, N) if multiple points are provided, or (2,) for a single point
    coords = np.vstack([x, y])
    
    # Apply the rotation using matrix multiplication (dot product)
    rotated_coords = rotation_matrix @ coords
    
    # Extract the new x and y coordinates
    new_x = rotated_coords[0]
    new_y = rotated_coords[1]
    
    return new_x, new_y

def hex_vertices(x, y, size=1.0, orientation='flat'):
    """
    Returns an (N, 6, 2) array of vertex coordinates for each hex center.

    Parameters
    ----------
    x, y : array-like
        Hex center coordinates (same length).
    size : float
        Radius of the hex (distance from center to vertex).
    orientation : {'pointy', 'flat'}
        Orientation of the hexagons.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if orientation == 'pointy':
        # vertices start at 30° and go around every 60°
        angles = np.deg2rad(np.arange(0, 360, 60)) + np.deg2rad(30)
    elif orientation == 'flat':
        # vertices start at 0° and go around every 60°
        angles = np.deg2rad(np.arange(0, 360, 60))
    else:
        raise ValueError("orientation must be 'pointy' or 'flat'")
    # Compute unit hexagon offsets
    dx = size * np.cos(angles)
    dy = size * np.sin(angles)
    
    # Add to every center (broadcasting)
    verts = np.stack([x[:, None] + dx, y[:, None] + dy], axis=-1)
    return verts


def pixel_from_pointy_hex(p,q):
    # https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    x = SQRT_3 * p + HALF_SQRT_3  * q
    y = 1.5 * q
    return x,y

def pixel_from_flat_hex(p,q):
    # https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    x =  1.5 * q                    
    y = HALF_SQRT_3 * q  +  SQRT_3 * p
    return x,y

def flat_hex_from_pixel(x, y, size=1.0):
    # https://www.redblobgames.com/grids/hexagons/#pixel-to-hex
    x /= size
    y /= size
    p = (2.0 / 3.0) * x
    q = (-1.0 / 3.0) * x + (3**0.5 / 3.0) * y
    return p,q

# --- Helpers ---
def cart_from_sph(az, el):
    # az: horizontal angle (radians), el: vertical angle (radians)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    # x points forward; y right; z up
    x = ce * ca
    y = ce * sa
    z = se
    return np.stack([x, y, z], axis=-1)

def sph_from_cart(x,y,z):
    az = np.arctan2(y, x)  # horizontal visual angle (right/left)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))  # vertical visual angle (up/down)
    return az, el

def unit_vectors_from_azel(az_el_deg):
    az = az_el_deg[:, 0]
    el = az_el_deg[:, 1]
    v = cart_from_sph(az, el)
    # already unit length from construction; normalize for safety:
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def angular_distance(U, V):
    # U: (N,3), V: (P,3). Returns (N,P) angles in radians
    # arccos of clipped dot products is stable
    dots = U @ V.T  # (N,P)
    np.clip(dots, -1.0, 1.0, out=dots)
    return np.arccos(dots)

def orthographic_fisheye_rays(image_shape, f, center=None, degrees=False):
    """
    Compute 3D ray directions for each pixel in an orthographic fisheye image.

    Parameters
    ----------
    image_shape : tuple
        (H, W) of the image.
    f : float
        Focal length in pixels (r = f * sin(theta)).
    center : tuple or None
        (x_c, y_c) image center. Defaults to center of the image.
    degrees : bool
        If True, returns azimuth/elevation maps in degrees.

    Returns
    -------
    rays : ndarray (H, W, 3)
        3D unit direction vector for each pixel.
    az_map : ndarray (H, W)
        Azimuth angle (atan2 of X,Z).
    el_map : ndarray (H, W)
        Elevation angle (atan2 of Y, sqrt(X^2 + Z^2)).
    """
    H, W = image_shape
    if center is None:
        x_c = (W - 1) / 2.0
        y_c = (H - 1) / 2.0
    else:
        x_c, y_c = center

    # Pixel grid
    y, x = np.indices((H, W))
    x = x - x_c
    y = y - y_c

    # Radius from optical center
    r = np.sqrt(x**2 + y**2)
    r_clamped = np.clip(r / f, 0, 1)

    # Polar coordinates
    theta = np.arcsin(r_clamped)           # polar angle from optical axis
    phi = np.arctan2(y, x)                 # azimuth around optical axis

    # 3D unit direction vectors
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    # Stack into HxWx3
    rays = np.stack([X, Y, Z], axis=-1)

    # Normalize (numerical safety)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Azimuth and elevation maps 
    az = np.arctan2(X, Z)
    el = -np.arctan2(Y, np.sqrt(Z**2 + X**2))

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    return az, el

def orthographic_fisheye_to_rectilinear_trimmed(img, cx, cy, f_pix, max_fov_deg=80):
    """
    Convert orthographic fisheye image to rectilinear projection.
    
    Properly handles offset lens centers by accounting for asymmetric FOV.
    
    Parameters
    ----------
    img : ndarray
        Input fisheye image (H, W) or (H, W, C)
    cx : float
        Principal point x-coordinate (column) in pixels - optical center
    cy : float
        Principal point y-coordinate (row) in pixels - optical center
    f_pix : float
        Focal length in pixels for orthographic fisheye (r = f * sin(theta))
    max_fov_deg : float
        Maximum field of view in degrees for output rectilinear image
    """
    # Convert FOV to radians
    theta_max = np.deg2rad(max_fov_deg)
    
    # Rectilinear projection: R = f * tan(theta)
    R_max = f_pix * np.tan(theta_max)
    R_max = abs(R_max)
    
    # Output region is +/- R_max in x and y
    out_half = int(np.ceil(R_max))
    out_w = 2 * out_half
    out_h = 2 * out_half
    
    # Define rectilinear grid in physical camera coordinates
    # This grid is centered at (0,0) in rectilinear image space
    x_rect = np.linspace(-R_max, R_max, out_w)
    y_rect = np.linspace(-R_max, R_max, out_h)
    xv, yv = np.meshgrid(x_rect, y_rect)
    
    # Rectilinear radius from center (in rectilinear image space)
    R = np.sqrt(xv**2 + yv**2)
    R_safe = np.maximum(R, 1e-9)
    
    # Rectilinear → theta (angle from optical axis)
    theta = np.arctan(R_safe / f_pix)
    
    # Theta → orthographic fisheye radius
    # For orthographic fisheye: r = f * sin(theta)
    r = f_pix * np.sin(theta)
    
    # Preserve direction (unit vector)
    # uf and vf are coordinates relative to the optical center (cx, cy)
    uf = r * (xv / R_safe)
    vf = r * (yv / R_safe)
    
    # Map to fisheye pixel coordinates
    # Add cx, cy to convert from optical center coordinates to pixel coordinates
    # In OpenCV: map_x corresponds to columns (x), map_y to rows (y)
    map_x = (uf + cx).astype(np.float32)
    map_y = (vf + cy).astype(np.float32)
    
    # Remap using OpenCV
    out = cv2.remap(img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return out


def read_flow_file(filename):
    with open(filename, 'rb') as f:
        # Check for the magic number
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print(f'Magic number incorrect. Invalid .flo file: {filename}')
            return None
        else:
            # Read the width and height
            w, h = np.fromfile(f, np.int32, count=2)
            # Read all the flow data (2 channels for u, v)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape the data into a 3D array (height, width, 2 channels)
            flow = np.resize(data, (h, w, 2))
            return flow

