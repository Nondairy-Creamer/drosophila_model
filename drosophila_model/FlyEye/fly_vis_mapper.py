from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from .utils import angular_distance, unit_vectors_from_azel
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
fly_data = DATA_DIR / "fly"
camera_data = DATA_DIR / "camera"

class FlyVisMapper:
    """
    Maps camera pixel responses to fly ommatidial responses using angular distance weighting.
    
    This class creates a mapping between camera pixels and fly ommatidia based on their
    angular positions in spherical coordinates. It uses Gaussian weighting to distribute
    pixel responses to nearby ommatidia based on angular distance.
    
    Attributes:
        camera_azimuth (np.ndarray): Azimuth angle map for camera pixels (H, W)
        camera_elevation (np.ndarray): Elevation angle map for camera pixels (H, W)
        fly_azimuth (np.ndarray): Azimuth angles for fly ommatidia (N,)
        fly_elevation (np.ndarray): Elevation angles for fly ommatidia (N,)
        p (np.ndarray): Hexagonal grid p-coordinates for ommatidia (N,)
        q (np.ndarray): Hexagonal grid q-coordinates for ommatidia (N,)
        W (scipy.sparse.csr_matrix): Weight matrix mapping pixels to ommatidia (N, P)
    
    Example:
        >>> from FlyEye.fly_vis_mapper import FlyVisMapper
        >>> import cv2
        >>> mapper = FlyVisMapper(fly_eye="both")
        >>> img = cv2.imread("image.jpg")
        >>> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        >>> responses = mapper.ommatidial_response(gray)
    """
    def __init__(self, fly_eye="both", camera_calibration_file=None, camera_azimuth=None, camera_elevation=None, center_azimuth=False):
        if camera_calibration_file is None:
            camera_calibration_file = camera_data / "two_camera_calibration.npz"
        if fly_eye == "both":
            fly_data_file = fly_data / "both_eyes.npz"
        elif fly_eye == "right":
            fly_data_file = fly_data / "right_eye.npz"        
        elif fly_eye == "left":
            fly_data_file = fly_data / "left_eye.npz"    
        else:
            raise ValueError(f"fly_eye must be 'both', 'right', or 'left', but got {fly_eye}")
        if camera_azimuth is None or camera_elevation is None:
            with np.load(camera_calibration_file) as data:
                self.camera_azimuth = data['azimuth']
                self.camera_elevation = data['elevation']
        else:
            self.camera_azimuth = camera_azimuth
            self.camera_elevation = camera_elevation
        with np.load(fly_data_file) as data:
            self.fly_azimuth = data['azimuth']
            self.fly_elevation = data['elevation']
            self.p = data['p']
            self.q = data['q']
            if center_azimuth: # useful if doing single eye mapping
                self.fly_azimuth -= self.fly_azimuth.mean()
        # --- Build weight matrix (dense or sparse) ---
        self.make_weight_matrix()
        print(f"Weight matrix shape: {self.W.shape}")
    
    def make_weight_matrix(self, sigma_deg=2.12, sparse=True, k_trunc=3.0, zero_row_value=None, return_zero_mask=False):
        """
        Builds a weight matrix mapping camera pixels to ommatidia using Gaussian weighting.
        
        Uses angular distance on the unit sphere to compute weights. Pixels are mapped to
        ommatidia based on their angular proximity, with weights following a Gaussian distribution.
        
        Parameters:
            sigma_deg (float): Standard deviation of Gaussian kernel in degrees (default: 2.12)
            sparse (bool): Use sparse matrix representation (default: True, recommended for efficiency)
            k_trunc (float): Truncation distance in multiples of sigma (default: 3.0)
            zero_row_value (float, optional): Value to assign ommatidia with no matching pixels.
                If None, uses nearest neighbor assignment.
            return_zero_mask (bool): If True, return boolean mask of zero rows (default: False)
        
        Returns:
            scipy.sparse.csr_matrix or np.ndarray: Weight matrix W of shape (N, P) where
                N is number of ommatidia and P is number of pixels.
                If return_zero_mask=True, returns tuple (W, zero_mask).
        
        Note:
            The weight matrix is row-normalized so that each ommatidium receives a weighted
            sum of pixel responses that sums to 1.0 (unless zero_row_value is specified).
        """
        omm_az_el_deg = np.column_stack((self.fly_azimuth,self.fly_elevation))
        pix_az_el_deg = np.column_stack((self.camera_azimuth.flatten(),self.camera_elevation.flatten()))
        
        # Identify invalid pixels (NaN/Inf) and exclude them from computation
        invalid_mask = ~np.isfinite(pix_az_el_deg).all(axis=1)
        valid_mask = ~invalid_mask
        valid_pixel_indices = np.where(valid_mask)[0]  # Indices of valid pixels in original array
        
        # Store invalid mask for later use
        self.invalid_pixel_mask = invalid_mask
        
        # Check if we have any valid pixels
        if len(valid_pixel_indices) == 0:
            raise ValueError("No valid pixels found in camera calibration. All pixels are NaN/Inf.")
        
        # Only use valid pixels for weight matrix computation
        if np.any(invalid_mask):
            pix_az_el_deg_valid = pix_az_el_deg[valid_mask]
        else:
            pix_az_el_deg_valid = pix_az_el_deg
        
        U = unit_vectors_from_azel(omm_az_el_deg)   # (N,3)
        V_valid = unit_vectors_from_azel(pix_az_el_deg_valid)   # (P_valid,3) - only valid pixels
        sigma = np.deg2rad(sigma_deg)
        N = U.shape[0]
        P_valid = V_valid.shape[0]
        P = pix_az_el_deg.shape[0]  # Total number of pixels (including invalid)
        trunc_angle = k_trunc * sigma
        
        if not sparse:
            # Dense mode: compute full matrix using only valid pixels
            theta_valid = angular_distance(U, V_valid)  # (N, P_valid)
            W_valid = np.exp(-0.5 * (theta_valid / sigma)**2, dtype=np.float64)
            # optional truncation for numerical cleanliness
            if k_trunc is not None:
                W_valid[theta_valid > trunc_angle] = 0.0
            
            # Create full matrix (N, P) with zeros for invalid pixels
            W = np.zeros((N, P), dtype=np.float64)
            W[:, valid_pixel_indices] = W_valid
            
            # row-normalize to sum to 1; handle rows that sum to 0
            row_sums = W.sum(axis=1, keepdims=True)
            zero_rows = (row_sums[:, 0] == 0)
            row_sums[~zero_rows] = row_sums[~zero_rows]
            W[~zero_rows] /= row_sums[~zero_rows]
            # For any zero rows (no pixels within truncation), fall back to nearest valid pixel
            # unless zero_row_value is specified
            if np.any(zero_rows):
                if zero_row_value is None:
                    # Connect to nearest valid pixel only
                    theta_all_valid = angular_distance(U[zero_rows], V_valid)
                    j_valid = theta_all_valid.argmin(axis=1)
                    j_original = valid_pixel_indices[j_valid]  # Map back to original pixel indices
                    W[zero_rows, :] = 0.0
                    W[zero_rows, j_original] = 1.0
                    zero_mask = np.zeros(N, dtype=bool) if return_zero_mask else None
                else:
                    # Leave as zero rows, compute mask if requested
                    zero_mask = np.zeros(N, dtype=bool) if return_zero_mask else None
                    if return_zero_mask:
                        zero_mask[zero_rows] = True
            else:
                zero_mask = np.zeros(N, dtype=bool) if return_zero_mask else None
            W = W.astype(np.float32)
            self.W = W
            return (W, zero_mask) if return_zero_mask else W
        
        # Sparse mode: use KD-tree for fast neighbor search
        # Build KD-tree on unit vectors (3D points on unit sphere) - only valid pixels
        tree = cKDTree(V_valid)
        
        # For angular distance on unit sphere, we can use Euclidean distance in 3D
        # as a proxy: chord distance = 2*sin(angle/2) ≈ angle for small angles
        # For trunc_angle, the chord distance threshold is approximately 2*sin(trunc_angle/2)
        # For better accuracy, we use the exact chord distance: 2*sin(angle/2)
        chord_threshold = 2.0 * np.sin(trunc_angle / 2.0)
        
        # Query all neighbors within chord distance
        # This is much faster than computing all pairwise distances
        rows, cols, data = [], [], []
        
        # Process in chunks to manage memory for very large N
        chunk_size = 10000
        for i0 in range(0, N, chunk_size):
            i1 = min(N, i0 + chunk_size)
            U_chunk = U[i0:i1]
            
            # Find all neighbors within chord distance (indices are into V_valid)
            # query_ball_point returns list of arrays/lists
            neighbors = tree.query_ball_point(U_chunk, chord_threshold)
            
            # Build sparse entries for this chunk - vectorized where possible
            chunk_rows, chunk_cols, chunk_data = [], [], []
            for local_i, neighbor_indices in enumerate(neighbors):
                global_i = i0 + local_i
                if len(neighbor_indices) == 0:
                    # No neighbors found - will handle zero rows later
                    continue
                
                neighbor_indices_valid = np.asarray(neighbor_indices, dtype=np.int32)
                neighbor_vecs = V_valid[neighbor_indices_valid]
                
                # Vectorized dot product computation
                dots = np.dot(U_chunk[local_i], neighbor_vecs.T)  # (len(neighbor_indices),)
                np.clip(dots, -1.0, 1.0, out=dots)
                theta_neighbors = np.arccos(dots)
                
                # Apply Gaussian weights
                w_neighbors = np.exp(-0.5 * (theta_neighbors / sigma)**2, dtype=np.float64)
                
                # Map valid pixel indices back to original pixel indices
                neighbor_indices_original = valid_pixel_indices[neighbor_indices_valid]
                
                # Add to chunk lists
                chunk_rows.append(np.full(len(neighbor_indices_original), global_i, dtype=np.int32))
                chunk_cols.append(neighbor_indices_original)
                chunk_data.append(w_neighbors)
            
            # Concatenate chunk results
            if chunk_rows:
                rows.append(np.concatenate(chunk_rows))
                cols.append(np.concatenate(chunk_cols))
                data.append(np.concatenate(chunk_data))
        
        # Handle case where no data was found
        if len(data) == 0:
            if zero_row_value is None:
                # Degenerate: connect each ommatidium to its nearest valid pixel
                distances, indices_valid = tree.query(U, k=1)
                indices_original = valid_pixel_indices[indices_valid.ravel()]  # Map to original indices
                W = coo_matrix((np.ones(N, dtype=np.float32),
                            (np.arange(N), indices_original)), shape=(N, P)).tocsr()
                self.W = W
                if return_zero_mask:
                    zero_mask = np.zeros(N, dtype=bool)
                    return W, zero_mask
                return W
            else:
                # Return empty matrix - all rows will be zero
                W = csr_matrix((N, P), dtype=np.float32)
                self.W = W
                if return_zero_mask:
                    zero_mask = np.ones(N, dtype=bool)  # All rows are zero
                    return W, zero_mask
                return W
        
        # Concatenate all chunks
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        
        # Build sparse matrix
        W = coo_matrix((data, (rows, cols)), shape=(N, P)).tocsr()
        
        # Row-normalize
        row_sums = np.array(W.sum(axis=1)).ravel()
        
        # Fix zero rows by connecting to nearest valid pixel with weight 1 (unless zero_row_value is set)
        zero_rows = np.where(row_sums == 0)[0]
        if zero_rows.size:
            if zero_row_value is None:
                # Connect to nearest valid pixel only
                distances, indices_valid = tree.query(U[zero_rows], k=1)
                indices_original = valid_pixel_indices[indices_valid.ravel()]  # Map to original indices
                W += coo_matrix((np.ones_like(indices_original, dtype=np.float64),
                                (zero_rows, indices_original)), shape=(N, P)).tocsr()
                row_sums = np.array(W.sum(axis=1)).ravel()
            # else: leave as zero rows, will be handled in ommatidial_response
        
        # Normalize rows
        if zero_row_value is None:
            # All rows should be non-zero now (either had neighbors or connected to nearest)
            inv = 1.0 / row_sums
            W = csr_matrix(W.multiply(inv[:, None]), dtype=np.float32)
            self.W = W
        else:
            # Only normalize non-zero rows (zero rows will be handled in ommatidial_response)
            non_zero_rows = np.where(row_sums > 0)[0]
            if non_zero_rows.size > 0:
                inv = np.zeros(N, dtype=np.float64)
                inv[non_zero_rows] = 1.0 / row_sums[non_zero_rows]
                W = csr_matrix(W.multiply(inv[:, None]), dtype=np.float32)
            else:
                # All rows are zero
                W = csr_matrix((N, P), dtype=np.float32)
            self.W = W
    def ommatidial_response(self, img, zero_row_value=None, zero_mask=None):
        """
        Maps an image to ommatidial responses using the precomputed weight matrix.
        
        Parameters:
            img (np.ndarray): Input image as numpy array of shape (H, W) for grayscale
                or (H, W, C) for multi-channel images.
            zero_row_value (float, optional): Value to assign ommatidia with no matching pixels.
                If None, uses the weight matrix default (nearest neighbor).
            zero_mask (np.ndarray, optional): Precomputed boolean mask of zero rows for
                performance optimization. If None, will be computed if needed.
        
        Returns:
            np.ndarray: Ommatidial responses as array of shape (N,) for grayscale images
                or (N, C) for multi-channel images, where N is the number of ommatidia
                and C is the number of channels.
        
        Example:
            >>> mapper = FlyVisMapper(fly_eye="both")
            >>> img = cv2.imread("image.jpg")
            >>> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            >>> responses = mapper.ommatidial_response(gray)
            >>> print(f"Responses shape: {responses.shape}")  # (N,)
        """
        H, Wimg = img.shape[:2]
        P = H * Wimg
        
        # Reshape and convert to float32 if needed
        if img.ndim == 2:
            if img.dtype != np.float32 or not img.flags['C_CONTIGUOUS']:
                stim = img.reshape(P).astype(np.float32, copy=False)
            else:
                stim = img.reshape(P)
            resp = self.W @ stim  # (N,)
        else:
            C = img.shape[2]
            if img.dtype != np.float32 or not img.flags['C_CONTIGUOUS']:
                stim = img.reshape(P, C).astype(np.float32, copy=False)
            else:
                stim = img.reshape(P, C)
            resp = self.W @ stim  # (N, C)
        
        # Handle zero rows if zero_row_value is specified
        if zero_row_value is not None:
            # Use precomputed mask if provided, otherwise compute it efficiently
            if zero_mask is None:
                # Fast check: count non-zero entries per row (much faster than sum)
                zero_mask = (self.W.getnnz(axis=1) == 0)
            
            if resp.ndim == 1:
                resp[zero_mask] = zero_row_value
            else:
                resp[zero_mask, :] = zero_row_value
        
        return resp
    
    def ommatidial_pixel_locations(self):
        """
        Computes the reverse mapping: for each ommatidium, returns the weighted average
        pixel location in (x, y) coordinates based on the weight matrix.
        
        Uses the Gaussian weights from the weight matrix to compute a weighted average
        of pixel coordinates for each ommatidium.
        
        Returns:
            np.ndarray: Array of shape (N, 2) where each row contains [x, y] pixel coordinates
                for the corresponding ommatidium. Coordinates are in pixel space where
                x corresponds to column index and y corresponds to row index.
        
        Example:
            >>> mapper = FlyVisMapper(fly_eye="both")
            >>> pixel_locs = mapper.ommatidial_pixel_locations()
            >>> print(f"Pixel locations shape: {pixel_locs.shape}")  # (N, 2)
            >>> print(f"Ommatidium 0 maps to pixel ({pixel_locs[0, 0]:.2f}, {pixel_locs[0, 1]:.2f})")
        """
        if not hasattr(self, 'W') or self.W is None:
            raise ValueError("Weight matrix W not found. Call make_weight_matrix() first.")
        
        # Get image dimensions from camera calibration
        H, W = self.camera_azimuth.shape
        
        # Create coordinate arrays for all pixels
        # Pixel index p corresponds to row p//W and column p%W
        # We want x (column) and y (row) coordinates
        P = H * W
        pixel_indices = np.arange(P, dtype=np.float32)
        x_coords = pixel_indices % W  # Column (x) coordinates
        y_coords = pixel_indices // W  # Row (y) coordinates
        
        # Stack into (P, 2) array: [x, y] for each pixel
        pixel_coords = np.column_stack((x_coords, y_coords))  # (P, 2)
        
        # Compute weighted average for each ommatidium
        # W has shape (N, P), pixel_coords has shape (P, 2)
        # Result: (N, 2) where each row is weighted average [x, y]
        if isinstance(self.W, csr_matrix):
            # For sparse matrices, use matrix multiplication
            # W @ pixel_coords gives (N, 2) weighted sum
            weighted_coords = self.W @ pixel_coords  # (N, 2)
        else:
            # For dense matrices
            weighted_coords = self.W @ pixel_coords  # (N, 2)
        
        return weighted_coords