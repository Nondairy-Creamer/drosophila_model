"""Hexagonal-lattice geometry for drawing optic-lobe columns as a hex mosaic."""
import numpy as np

SQRT_3 = np.sqrt(3)
HALF_SQRT_3 = SQRT_3/2

# Rotation (degrees) applied to the flat-hex layout so the mosaic is drawn
# upright; the layout is rotated by -ROTATE_CORRECTION.
ROTATE_CORRECTION = 60


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


def pixel_from_flat_hex(p,q):
    # https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    x =  1.5 * q
    y = HALF_SQRT_3 * q  +  SQRT_3 * p
    return x,y


def column_hex_centers(column_p_q):
    """
    Hex-mosaic centers ``(x, y)`` for optic-lobe column lattice coordinates.

    ``column_p_q`` is the ``(N, 2)`` connectome (p, q) per column
    (``model.regions['optic_lobe'].column_p_q``, from
    ``connectome_data/baked/optic_lobe_columns.parquet``). q is negated, the
    columns are laid out on a flat-topped hex grid (unit hex radius) and the
    layout is rotated by ``-ROTATE_CORRECTION`` degrees. This is the single
    definition of the mosaic layout used by the ``--hex`` animations and the
    diamond movies.
    """
    p = np.asarray(column_p_q[:, 0])
    q = -np.asarray(column_p_q[:, 1])
    x, y = pixel_from_flat_hex(p, q)
    return rotate_coords(x, y, -ROTATE_CORRECTION)
