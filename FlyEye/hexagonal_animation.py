import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

# Handle imports for both script execution and module import
try:
    from .utils import rotate_coords, pixel_from_flat_hex, hex_vertices
    from .fly_vis_mapper import FlyVisMapper
    ROOT = Path(__file__).resolve().parents[1]
except ImportError:
    src_path = Path(__file__).resolve().parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from FlyEye.utils import rotate_coords, pixel_from_flat_hex, hex_vertices
    from FlyEye.fly_vis_mapper import FlyVisMapper
    ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
fly_data = DATA_DIR / "fly"
camera_data = DATA_DIR / "camera"
ROTATE_CORRECTION = 60


def color_hexagons(p, q, values, plot=True, filename=None, colormap=None, fps=20,
                   titles=None, vmin=None, vmax=None, colorbar=True):
    """
    Creates animated visualization of ommatidial responses as colored hexagons.
    
    Parameters:
        p (np.ndarray): Hexagonal grid p-coordinates (N,)
        q (np.ndarray): Hexagonal grid q-coordinates (N,)
        values (np.ndarray): Response values per ommatidium, shape (T, N) where T is number of frames.
            Values should be integer indices into *colormap* (0-255).
        plot (bool): Display plot interactively (default: True)
        filename (str, optional): Output video filename (e.g., "output.mp4")
        colormap (np.ndarray, optional): Matplotlib colormap array (256, 4). If None, uses plasma.
        fps (int): Frames per second for output video (default: 20)
        titles (list[str], optional): Per-frame titles.
        vmin (float, optional): Data value corresponding to colormap index 0.
        vmax (float, optional): Data value corresponding to colormap index 255.
        colorbar (bool): Show a colorbar alongside the hexagons (default: True).
    
    Returns:
        None: Displays animation and optionally saves to file.
    """
    from matplotlib.colors import Normalize, ListedColormap
    import matplotlib.colorbar as mcolorbar

    x,y = pixel_from_flat_hex(p,q)
    rot_x, rot_y=rotate_coords(x, y, -ROTATE_CORRECTION)
    verts = hex_vertices(rot_x,rot_y)
    if colormap is None:
        colormap = cm.plasma(np.linspace(0.0, 1, 256))
    # setup animation
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect('equal')
    polys = []
    for vert in verts:
        closed = np.concatenate((vert, vert[:1, :]), axis=0)
        p_poly = ax.fill(closed[:, 0], closed[:, 1], color='gray')[0]
        polys.append(p_poly)

    if colorbar and vmin is not None and vmax is not None:
        listed_cmap = ListedColormap(colormap[:, :3])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    # Animation update function
    def update(frame):
        curr_values = values[frame]
        for it, p_poly in enumerate(polys):
            color = colormap[int(curr_values[it]),:3]
            p_poly.set_facecolor(color)
        if titles is not None:
            ax.set_title(titles[frame])
        return polys
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(values), interval=1/fps*1000, blit=False)
    if filename is not None:
        anim.save(
            filename,
            writer="ffmpeg",
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"]
        )
    if plot:
        plt.show()


def hexagonal_from_video(dir_to_process, fly_eye='both', filename=None, camera_azimuth=None, camera_elevation=None, plot=True):
    """
    Process directory of images and create hexagonal visualization of fly vision.
    
    Parameters:
        dir_to_process (str or Path): Directory containing image files to process
        fly_eye (str): Which eye to use - "both", "right", or "left" (default: "both")
        filename (str, optional): Output video filename (e.g., "output.mp4")
        camera_azimuth (np.ndarray, optional): Camera azimuth angle map (overrides calibration file)
        camera_elevation (np.ndarray, optional): Camera elevation angle map (overrides calibration file)
        plot (bool): Display plot interactively (default: True)
    
    Returns:
        None: Creates visualization and optionally saves to file.
    
    Raises:
        ValueError: If no image files found or invalid fly_eye parameter.
    """
    if fly_eye == "both":
        camera_calibration_file = camera_data / 'two_camera_calibration.npz'
        fly_vis_mapper = FlyVisMapper(camera_calibration_file=camera_calibration_file,fly_eye='both',camera_azimuth=camera_azimuth,camera_elevation=camera_elevation)
    elif fly_eye == "right":
        camera_calibration_file = camera_data / 'cam0_calibration.npz'     
        fly_vis_mapper = FlyVisMapper(camera_calibration_file=camera_calibration_file,fly_eye='right',camera_azimuth=camera_azimuth,camera_elevation=camera_elevation)
    elif fly_eye == "left":
        camera_calibration_file = camera_data / 'cam1_calibration.npz'
        fly_vis_mapper = FlyVisMapper(camera_calibration_file=camera_calibration_file,fly_eye='left',camera_azimuth=camera_azimuth,camera_elevation=camera_elevation)
    else:
        raise ValueError(f"fly_eye must be both, right or left, but got {fly_eye}")
    
    # Filter to only image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(dir_to_process, ext)))
        files.extend(glob.glob(os.path.join(dir_to_process, ext.upper())))
    files = natsorted(files)

    if len(files) == 0:
        raise ValueError(f"No image files found in directory: {dir_to_process}")
    
    responses = [] 
    for file in files:
        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Could not read image file: {file}, skipping...")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        responses.append(fly_vis_mapper.ommatidial_response(gray))

    if len(responses) == 0:
        raise ValueError(f"No valid images could be processed from directory: {dir_to_process}")
    
    responses = np.array(responses)
    color_hexagons(fly_vis_mapper.p,fly_vis_mapper.q,responses,plot=plot,filename=filename,colormap=None,fps=20)

def main():
    """Entry point for the console script."""
    parser = argparse.ArgumentParser(
        description='Process directory of images and create hexagonal visualization of fly vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  hexagonal-from-video video_frames/
  
  # Use right eye and save to file
  hexagonal-from-video video_frames/ --fly-eye right --filename right_eye.mp4
  
  # Use left eye without plotting
  hexagonal-from-video video_frames/ --fly-eye left --filename left_eye.mp4 --no-plot
  
  # With custom camera angles
  hexagonal-from-video video_frames/ --camera-azimuth 45.0 --camera-elevation 30.0
        """
    )
    
    parser.add_argument(
        'dir_to_process',
        type=str,
        help='Directory containing image files to process (supports: .jpg, .jpeg, .png, .bmp, .tiff, .tif)'
    )
    parser.add_argument(
        '--fly-eye',
        type=str,
        choices=['both', 'left', 'right'],
        default='both',
        help='Which fly eye(s) to use (default: both)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Output filename for video (e.g., output.mp4). If not specified, video is not saved'
    )
    parser.add_argument(
        '--camera-azimuth',
        type=float,
        default=None,
        help='Camera azimuth angle in degrees (overrides calibration file)'
    )
    parser.add_argument(
        '--camera-elevation',
        type=float,
        default=None,
        help='Camera elevation angle in degrees (overrides calibration file)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable interactive plotting (default: plot is enabled)'
    )
    
    args = parser.parse_args()
    
    hexagonal_from_video(
        dir_to_process=args.dir_to_process,
        fly_eye=args.fly_eye,
        filename=args.filename,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        plot=not args.no_plot
    )


if __name__ == "__main__":
    main()


