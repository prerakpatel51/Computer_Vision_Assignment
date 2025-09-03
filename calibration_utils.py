# calibration_utils.py
# Core helpers for OpenCV camera calibration with clean, stateless design.

from __future__ import annotations  # Enable forward type hints
import os, glob, json  # File operations, pattern matching, JSON handling
from typing import List, Tuple, Dict, Any  # Type annotations

import numpy as np  # Numerical operations and array handling
import cv2  # Computer vision library for image processing and calibration
import plotly.graph_objects as go  # Interactive 3D plotting
import matplotlib.pyplot as plt  # Static plotting library


# ---------- IO helpers ----------
class IO:
    @staticmethod
    def ensure_dir(path: str) -> None:
        """Create directory if it doesn't exist, including parent directories"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def list_images(dir_path: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")) -> List[str]:
        """Find all image files in directory with specified extensions"""
        files: List[str] = []  # Initialize empty list to store found files
        for e in exts:  # Loop through each file extension
            # Use glob to find all files with current extension
            files.extend(glob.glob(os.path.join(dir_path, f"*{e}")))
        return sorted(files)  # Return alphabetically sorted list

    @staticmethod
    def imread_rgb(path: str):
        """Read image file and convert from BGR (OpenCV default) to RGB format"""
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # Load image in color mode
        if bgr is None:  # Check if image loading failed
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    @staticmethod
    def imwrite_rgb(path: str, rgb: np.ndarray) -> None:
        """Save RGB image by converting to BGR format (OpenCV's expected format)"""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imwrite(path, bgr)  # Save the BGR image

    @staticmethod
    def save_json(obj: Dict[str, Any], path: str) -> None:
        """Save Python object as formatted JSON file"""
        with open(path, "w") as f:  # Open file in write mode
            json.dump(obj, f, indent=2)  # Write with 2-space indentation

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """Load JSON file and return as Python dictionary"""
        with open(path, "r") as f:  # Open file in read mode
            return json.load(f)  # Parse JSON and return dictionary


# ---------- Board model ----------
class Board:
    """
    Chessboard object model.
    pattern_size = (cols, rows) are INNER corners e.g. (9,6)
    square_size = side length in meters e.g. 0.025
    """
    @staticmethod
    def object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
        """Generate 3D object points for chessboard pattern in world coordinates"""
        cols, rows = pattern_size  # Unpack pattern dimensions
        # Create array to hold 3D points (Z=0 for all points on flat board)
        objp = np.zeros((rows * cols, 3), np.float32)
        # Generate 2D grid coordinates for all corner intersections
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        # Scale grid by square size to get real-world coordinates
        objp[:, :2] = grid * square_size  # Assign X,Y coordinates, Z remains 0
        return objp  # Return array of 3D world coordinates


# ---------- Image helpers ----------
class Img:
    @staticmethod
    def to_gray(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using OpenCV"""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # Convert color space

    @staticmethod
    def find_corners(gray: np.ndarray, pattern_size: Tuple[int, int]):
        """Detect chessboard corners in grayscale image with sub-pixel refinement"""
        # Set flags for robust corner detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        # Attempt to find chessboard corners
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ok:  # If corner detection failed
            return False, None
        # Define termination criteria for sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        # Refine corner positions to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners_refined  # Return success and refined corners


# ---------- Calibration pipeline ----------
class Calib:
    @staticmethod
    def calibrate(image_paths: List[str], pattern_size=(9, 6), square_size=0.025) -> Dict[str, Any]:
        """Perform camera calibration and return intrinsics, distortion, and pose data"""
        obj_pts: List[np.ndarray] = []  # List to store 3D object points
        img_pts: List[np.ndarray] = []  # List to store 2D image points
        imsize = None  # Will store image dimensions
        objp = Board.object_points(pattern_size, square_size)  # Generate 3D board points

        valid_paths: List[str] = []  # Track which images had successful corner detection
        # Process each calibration image
        for p in image_paths:
            rgb = IO.imread_rgb(p)  # Load image in RGB format
            if rgb is None:  # Skip if image loading failed
                continue
            gray = Img.to_gray(rgb)  # Convert to grayscale for corner detection
            if imsize is None:  # Store image size from first valid image
                imsize = (gray.shape[1], gray.shape[0])  # (width, height)
            ok, corners = Img.find_corners(gray, pattern_size)  # Find chessboard corners
            if ok:  # If corners were successfully detected
                obj_pts.append(objp.copy())  # Add 3D world points
                img_pts.append(corners)  # Add corresponding 2D image points
                valid_paths.append(p)  # Track this image as valid

        # Check if we have enough valid detections for calibration
        if imsize is None or len(obj_pts) < 5:
            raise ValueError("Not enough valid detections (need at least ~5).")

        # Perform camera calibration using OpenCV
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=obj_pts,    # List of 3D world points
            imagePoints=img_pts,     # List of corresponding 2D image points
            imageSize=imsize,        # Image dimensions (width, height)
            cameraMatrix=None,       # Let OpenCV estimate camera matrix
            distCoeffs=None,         # Let OpenCV estimate distortion coefficients
            flags=0                  # Use default calibration flags
        )

        # Calculate reprojection error for each view
        per_view_errors = []
        for i in range(len(obj_pts)):
            # Project 3D points back to 2D using estimated camera parameters
            proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
            # Calculate RMS error between detected and projected points
            err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
            per_view_errors.append(float(err))  # Store error for this view
        mean_err = float(np.mean(per_view_errors))  # Average error across all views

        # Convert rotation vectors to rotation matrices for each view
        extrinsics = []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)  # Convert rotation vector to 3x3 matrix
            # Store rotation matrix and translation vector
            extrinsics.append({"R": R.tolist(), "t": tv.flatten().tolist()})

        # Package all calibration results into a dictionary
        result = {
            "ret_rms": float(ret),                    # Overall RMS reprojection error
            "mean_reproj_error": mean_err,            # Mean error across all views
            "image_size": {"width": imsize[0], "height": imsize[1]},  # Image dimensions
            "K": np.asarray(K).tolist(),              # Camera intrinsic matrix (3x3)
            "D": np.asarray(D).flatten().tolist(),    # Distortion coefficients
            "pattern_size": {"cols": pattern_size[0], "rows": pattern_size[1]},  # Board size
            "square_size_m": float(square_size),      # Physical square size in meters
            "num_views": len(valid_paths),            # Number of calibration images used
            "per_view_errors": per_view_errors,       # Individual view errors
            "valid_paths": valid_paths,               # Paths to successfully used images
            "extrinsics": extrinsics                  # Camera poses for each view
        }
        return result  # Return complete calibration data

    @staticmethod
    def undistort(rgb: np.ndarray, K, D, keep_size=True):
        """Remove lens distortion from image using calibrated camera parameters"""
        h, w = rgb.shape[:2]  # Get image height and width
        K = np.array(K, dtype=np.float64)  # Convert intrinsic matrix to numpy array
        D = np.array(D, dtype=np.float64)  # Convert distortion coefficients to numpy array
        # Calculate optimal new camera matrix for undistorted image
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)  # alpha=0 crops image
        # Apply undistortion using original and new camera matrices
        und = cv2.undistort(rgb, K, D, None, newK)
        if keep_size:  # If requested, resize back to original dimensions
            und = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
        return und  # Return undistorted image


# ---------- Overlay helpers ----------
class Overlay:
    @staticmethod
    def _draw_axes_at_point(rgb, K, D, rvec, tvec, origin3d, axis_len):
        """Draw 3D coordinate axes (X=red, Y=green, Z=blue) at specified 3D point"""
        # Define 3D points for origin and axis endpoints
        P = np.float32([
            origin3d,                        # Origin point
            origin3d + [axis_len, 0, 0],    # X-axis endpoint (red)
            origin3d + [0, axis_len, 0],    # Y-axis endpoint (green)
            origin3d + [0, 0, axis_len],    # Z-axis endpoint (blue)
        ]).reshape(-1, 3)
        # Project 3D points to 2D image coordinates
        pts2d, _ = cv2.projectPoints(P, rvec, tvec, K, D)
        pts2d = pts2d.reshape(-1, 2).astype(int)  # Convert to integer pixel coordinates

        img = rgb.copy()  # Create copy to avoid modifying original
        # Extract projected points for origin and each axis
        o = tuple(pts2d[0]); x = tuple(pts2d[1]); y = tuple(pts2d[2]); z = tuple(pts2d[3])
        # Draw colored lines for each axis
        cv2.line(img, o, x, (255, 0, 0), 3)   # X-axis: red
        cv2.line(img, o, y, (0, 255, 0), 3)   # Y-axis: green
        cv2.line(img, o, z, (0, 0, 255), 3)   # Z-axis: blue
        return img  # Return image with drawn axes

    @staticmethod
    def make_sample_overlays(
        image_paths, K, D, pattern_size, square_size,
        max_images=8, axis_scale=3.0, draw_corners=False, anchor="origin"
    ):
        """Create overlay images with 3D coordinate axes projected onto calibration images"""
        K = np.array(K, np.float64)  # Convert camera intrinsic matrix to numpy array
        D = np.array(D, np.float64)  # Convert distortion coefficients to numpy array
        cols, rows = pattern_size    # Unpack chessboard dimensions
        # Generate 3D object points for both orientations (in case board is rotated)
        objp_main = Board.object_points((cols, rows), square_size)  # Standard orientation
        objp_swap = Board.object_points((rows, cols), square_size)  # Swapped orientation

        # Define anchor points for coordinate axes placement
        center3d = np.array([(cols - 1) * square_size / 2.0,   # X center of chessboard
                             (rows - 1) * square_size / 2.0,   # Y center of chessboard
                             0.0], dtype=np.float32)           # Z=0 (board plane)
        origin3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Origin corner of board

        chosen = image_paths[:max_images]  # Limit number of processed images
        out = []  # List to store (path, overlay_image) tuples
        # Process each selected image
        for p in chosen:
            rgb = IO.imread_rgb(p)  # Load image in RGB format
            if rgb is None:         # Skip if image loading failed
                continue
            gray = Img.to_gray(rgb)  # Convert to grayscale for corner detection

            # Initialize variables to track successful detection
            used_corners = None  # Will store successfully detected corners
            rvec = tvec = None   # Will store camera pose (rotation and translation vectors)

            # Try detecting chessboard in standard orientation first
            ok, corners = Img.find_corners(gray, (cols, rows))
            if ok:  # If corners detected successfully
                # Solve for camera pose using PnP (Perspective-n-Point) algorithm
                ok2, rvec, tvec = cv2.solvePnP(objp_main, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok2:  # If pose estimation succeeded
                    used_corners = corners  # Store the successful corner detection

            # If standard orientation failed, try swapped dimensions
            if used_corners is None:
                okS, cornersS = Img.find_corners(gray, (rows, cols))  # Try swapped pattern
                if okS:  # If corners detected with swapped orientation
                    # Solve for camera pose using swapped object points
                    ok2, rvec, tvec = cv2.solvePnP(objp_swap, cornersS, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                    if ok2:  # If pose estimation succeeded
                        used_corners = cornersS  # Store the successful corner detection

            # Skip this image if corner detection or pose estimation failed
            if used_corners is None or rvec is None:
                continue

            # Choose where to place the coordinate axes based on user preference
            anchor_pt = origin3d if anchor == "origin" else center3d
            # Draw 3D coordinate axes projected onto the image
            img = Overlay._draw_axes_at_point(
                rgb, K, D, rvec, tvec, anchor_pt, 
                axis_len=square_size * float(axis_scale)  # Scale axes relative to square size
            )

            # Optionally draw detected chessboard corners as green circles
            if draw_corners:
                pts = used_corners.reshape(-1, 2)  # Reshape to list of (x,y) points
                for (x, y) in pts:  # Draw each corner
                    cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            out.append((p, img))  # Add (image_path, overlay_image) to results
        return out  # Return list of overlay images with projected axes


# ---------- 3D camera pose visualization ----------
class Render:
    @staticmethod
    def camera_centers_and_axes(extrinsics: List[Dict[str, Any]]):
        """Extract camera centers and optical axis directions from extrinsic parameters"""
        centers = []  # List to store 3D camera center positions
        z_axes = []   # List to store optical axis directions (Z-axis of camera)
        
        # Process each camera pose from calibration results
        for ex in extrinsics:
            # Extract rotation matrix and translation vector
            R = np.array(ex["R"], dtype=np.float64)     # 3x3 rotation matrix
            t = np.array(ex["t"], dtype=np.float64).reshape(3, 1)  # 3x1 translation vector
            # Convert from camera-to-world transform to world camera center
            C = (-R.T @ t).flatten()                    # Camera center in world coordinates
            # Get camera's optical axis direction (Z-axis) in world coordinates
            z_axis = (R.T @ np.array([0, 0, 1.0])).flatten()  # Z-axis direction vector
            centers.append(C); z_axes.append(z_axis)    # Store results
        return np.array(centers), np.array(z_axes)      # Return as numpy arrays

    @staticmethod
    def plot_poses_plotly(extrinsics: List[Dict[str, Any]], square_size=0.025, board_size=(9, 6)):
        """Create interactive 3D plot of camera poses using Plotly"""
        # Extract camera positions and orientations
        centers, z_axes = Render.camera_centers_and_axes(extrinsics)
        cols, rows = board_size  # Unpack chessboard dimensions
        # Calculate physical board dimensions in meters
        w = (cols-1)*square_size; h = (rows-1)*square_size  # Width and height

        # Define chessboard outline as closed rectangle (Z=0 plane)
        bx = [0, w, w, 0, 0]; by = [0, 0, h, h, 0]; bz = [0, 0, 0, 0, 0]
        fig = go.Figure()  # Create new Plotly figure
        # Add chessboard outline as 3D line plot
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', name='Board'))
        
        if len(centers) > 0:  # If we have camera poses to visualize
            # Plot camera centers as 3D scatter points
            fig.add_trace(go.Scatter3d(x=centers[:,0], y=centers[:,1], z=centers[:,2],
                                       mode='markers', name='Cameras', marker=dict(size=5)))
            scale = square_size*3  # Scale factor for optical axis visualization
            # Draw optical axis for each camera as a line from center outward
            for c, z in zip(centers, z_axes):
                p2 = c + z*scale  # End point of optical axis line
                fig.add_trace(go.Scatter3d(x=[c[0], p2[0]], y=[c[1], p2[1]], z=[c[2], p2[2]],
                                           mode='lines', line=dict(width=4), name='optical_axis'))
        # Configure plot layout and appearance
        fig.update_layout(width=800, height=500,  # Set figure dimensions
                          scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                                     aspectmode="data"),  # Equal aspect ratio for all axes
                          title="Estimated Camera Poses w.r.t. Chessboard")  # Plot title
        return fig  # Return interactive Plotly figure

    @staticmethod
    def plot_poses_matplotlib(extrinsics: List[Dict[str, Any]], square_size=0.025, board_size=(9, 6)):
        """Create static 3D plot of camera poses using Matplotlib"""
        # Extract camera positions and orientations
        centers, z_axes = Render.camera_centers_and_axes(extrinsics)
        cols, rows = board_size  # Unpack chessboard dimensions
        # Calculate physical board dimensions in meters
        w = (cols-1)*square_size; h = (rows-1)*square_size  # Width and height

        # Create matplotlib figure with 3D axes
        fig = plt.figure(figsize=(7, 5))  # Set figure size
        ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot
        # Define chessboard outline coordinates
        bx = [0, w, w, 0, 0]; by = [0, 0, h, h, 0]; bz = [0, 0, 0, 0, 0]
        # Plot chessboard outline as 3D line
        ax.plot(bx, by, bz, lw=2, label="Board")
        
        if len(centers) > 0:  # If we have camera poses to visualize
            # Plot camera centers as 3D scatter points
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], s=10, label="Cameras")
            scale = square_size*3  # Scale factor for optical axis visualization
            # Draw optical axis for each camera
            for c, z in zip(centers, z_axes):
                p2 = c + z*scale  # End point of optical axis line
                ax.plot([c[0], p2[0]], [c[1], p2[1]], [c[2], p2[2]], lw=2)
        # Configure axes labels and plot appearance
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")  # Axis labels
        ax.legend(); ax.set_title("Estimated Camera Poses w.r.t. Chessboard")    # Legend and title
        plt.tight_layout()  # Optimize subplot spacing
        return fig  # Return matplotlib figure
