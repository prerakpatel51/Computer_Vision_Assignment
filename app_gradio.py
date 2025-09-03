# app_gradio.py - Gradio web interface for camera calibration

import os, glob, shutil  # File operations and directory management
import numpy as np       # Numerical operations for arrays
import gradio as gr      # Web interface framework

# Import custom calibration utilities
from calibration_utils import IO, Calib, Overlay, Render, Img

# Environment detection and directory setup
IS_COLAB = os.path.exists("/content")  # Check if running in Google Colab
BASE_DIR = "/content" if IS_COLAB else os.getcwd()  # Use /content in Colab, current dir elsewhere
IMG_DIR = os.path.join(BASE_DIR, "images")  # Directory for calibration images

# For samples, check both current directory and base directory
SAMPLES_DIR = os.path.join(os.getcwd(), "samples") if os.path.exists(os.path.join(os.getcwd(), "samples")) else os.path.join(BASE_DIR, "samples")

OUT_JSON = os.path.join(BASE_DIR, "calibration.json")  # Output file for calibration results
IO.ensure_dir(IMG_DIR)  # Create images directory if it doesn't exist

def load_sample_images():
    """Load sample images from samples/ folder to images/ folder on startup"""
    if not os.path.exists(SAMPLES_DIR):
        return "No samples/ folder found. Upload your own images to get started."
    
    # Get sample images
    sample_files = IO.list_images(SAMPLES_DIR)
    if not sample_files:
        return "No sample images found in samples/ folder. Upload your own images to get started."
    
    # Clear existing images from the directory
    for f in glob.glob(os.path.join(IMG_DIR, "*")):
        try: os.remove(f)
        except: pass
    
    # Copy sample images to images directory
    saved = 0
    for i, src in enumerate(sample_files):
        if os.path.exists(src):
            base = os.path.splitext(os.path.basename(src))[0] + ".jpg"
            shutil.copy(src, os.path.join(IMG_DIR, f"sample_{i:03d}_{base}"))
            saved += 1
    
    return f"✅ Loaded {saved} sample image(s) from samples/ folder. You can now run calibration or upload different images."

def save_uploads(files):
    """Save uploaded files to images directory, clearing existing files first"""
    # Clear existing images from the directory
    for f in glob.glob(os.path.join(IMG_DIR, "*")):
        try: os.remove(f)  # Remove each existing file
        except: pass       # Ignore errors if file can't be removed
    
    saved = 0  # Counter for successfully saved files
    if files:  # If files were uploaded
        for i, src in enumerate(files):  # Iterate through uploaded filepath strings
            if src and os.path.exists(src):  # Check if file path is valid and exists
                # Create standardized filename with .jpg extension
                base = os.path.splitext(os.path.basename(src))[0] + ".jpg"
                # Copy file to images directory with numbered prefix
                shutil.copy(src, os.path.join(IMG_DIR, f"calib_{i:03d}_{base}"))
                saved += 1  # Increment counter
    return f"Saved {saved} image(s) to {IMG_DIR}"  # Return status message

def test_detection(cols, rows):
    """Test chessboard corner detection on uploaded images without running calibration"""
    try:
        paths = IO.list_images(IMG_DIR)  # Get list of all images in directory
        pattern = (int(cols), int(rows))  # Convert to integer tuple for pattern size
        det_ok, missed = 0, []  # Counters for successful detections and missed images
        
        # Test corner detection on each image
        for p in paths:
            rgb = IO.imread_rgb(p)  # Load image in RGB format
            if rgb is None:         # Skip if image loading failed
                continue
            # Attempt to find chessboard corners in grayscale image
            ok, _ = Img.find_corners(Img.to_gray(rgb), pattern)
            det_ok += int(ok)       # Count successful detections
            if not ok: missed.append(os.path.basename(p))  # Track failed images
        
        # Build status message
        msg = [f"Pattern {pattern}: detected in {det_ok}/{len(paths)} images."]
        if missed: msg.append("Missed (first 10): " + ", ".join(missed[:10]))
        if det_ok < 5: msg.append(f"Try swapped pattern {(pattern[1], pattern[0])}.")
        return "\n".join(msg)  # Return formatted status report
    except Exception as e:
        # Return detailed error information if detection test fails
        import traceback; return f"Detection test failed:\n{e}\n\n{traceback.format_exc()}"

def run_calibration(cols, rows, square_size_m):
    """Execute full camera calibration and save results to JSON file"""
    try:
        paths = IO.list_images(IMG_DIR)  # Get all calibration images
        if len(paths) < 5:  # Check minimum image requirement
            return f"Only {len(paths)} images in {IMG_DIR}. Need ≥5."

        pattern = (int(cols), int(rows))      # Convert pattern size to integers
        square_size_m = float(square_size_m)  # Convert square size to float

        # Pre-check: count how many images have detectable corners
        det_ok = 0
        for p in paths:
            rgb = IO.imread_rgb(p)  # Load image
            if rgb is None: continue  # Skip failed loads
            # Test corner detection on this image
            ok, _ = Img.find_corners(Img.to_gray(rgb), pattern)
            det_ok += int(ok)  # Count successful detections
        
        # Abort if too few images have detectable corners
        if det_ok < 5:
            return (f"Detected corners in only {det_ok}/{len(paths)} images with pattern {pattern}. "
                    f"Try {(rows, cols)} or retake clearer shots.")

        # Run the actual calibration algorithm
        result = Calib.calibrate(paths, pattern_size=pattern, square_size=square_size_m)
        IO.save_json(result, OUT_JSON)  # Save results to JSON file

        # Extract camera parameters for display
        K = np.array(result["K"]); D = np.array(result["D"])  # Intrinsics and distortion
        
        # Custom formatting for clean matrix display (force no scientific notation)
        def format_matrix(matrix, precision=2):
            """Format matrix without scientific notation"""
            if matrix.ndim == 1:
                # Format 1D array (distortion coefficients)
                formatted_values = []
                for x in matrix:
                    if abs(x) < 1e-10:  # Handle very small numbers
                        formatted_values.append("0.00")
                    else:
                        formatted_values.append(f"{x:.{precision}f}")
                return "[" + ", ".join(formatted_values) + "]"
            else:
                # Format 2D matrix (intrinsic matrix)
                rows = []
                for row in matrix:
                    formatted_values = []
                    for x in row:
                        if abs(x) < 1e-10:  # Handle very small numbers
                            formatted_values.append("    0.00")
                        else:
                            formatted_values.append(f"{x:8.{precision}f}")
                    row_str = "[" + ", ".join(formatted_values) + "]"
                    rows.append(row_str)
                return "[\n " + ",\n ".join(rows) + "\n]"
        
        # Apply formatting with explicit precision
        K_formatted = format_matrix(K, precision=2)
        D_formatted = format_matrix(D, precision=4)
        
        # Build comprehensive result message
        msg = [
            f"Saved calibration to: {OUT_JSON}",
            f"Views used: {result['num_views']}",
            f"Image size: {result['image_size']}",
            f"RMS reprojection error: {result['ret_rms']:.4f}",
            f"Mean per-view error: {result['mean_reproj_error']:.4f}",
            f"K (intrinsics):\n{K_formatted}",
            f"D (distortion): {D_formatted}",
            "",
            "Tip: In Colab, open the Files panel (left sidebar) to download calibration.json.",
        ]
        return "\n".join(msg)  # Return formatted results
    except Exception as e:
        # Return detailed error information if calibration fails
        import traceback; return f"Calibration failed:\n{e}\n\n{traceback.format_exc()}"

def show_poses_and_overlays(max_overlays, use_plotly=True, axis_scale=3.0, draw_corners=False, anchor="origin"):
    """Generate 3D pose visualization, overlay images, and undistortion preview"""
    if not os.path.exists(OUT_JSON): return None, [], None  # Return empty if no calibration exists
    
    # Load calibration results from JSON file
    calib = IO.load_json(OUT_JSON)
    # Extract camera parameters
    K = np.array(calib["K"], float); D = np.array(calib["D"], float)  # Intrinsics & distortion
    # Extract chessboard configuration
    cols = int(calib["pattern_size"]["cols"]); rows = int(calib["pattern_size"]["rows"])
    square = float(calib["square_size_m"])  # Physical square size
    extr = calib.get("extrinsics", [])      # Camera poses for each view
    # Get image paths (from calibration or fallback to all images)
    paths = calib.get("valid_paths", []) or IO.list_images(IMG_DIR)

    # Generate 3D pose plot using either interactive (Plotly) or static (Matplotlib)
    fig = Render.plot_poses_plotly(extr, square_size=square, board_size=(cols, rows)) if use_plotly \
          else Render.plot_poses_matplotlib(extr, square_size=square, board_size=(cols, rows))
    
    # Create overlay images with 3D axes projected onto calibration images
    overlays = Overlay.make_sample_overlays(
        paths, K, D, (cols, rows), square,
        max_images=int(max_overlays),       # Limit number of overlay images
        axis_scale=float(axis_scale),       # Size of drawn axes
        draw_corners=bool(draw_corners),    # Whether to draw detected corners
        anchor=anchor,                      # Where to place axes (origin/center)
    )
    gallery = [img for (_p, img) in overlays]  # Extract images for gallery display
    
    # Create undistortion comparison (original | undistorted side-by-side)
    und = None
    if paths:  # If we have images to work with
        samp = IO.imread_rgb(paths[0])  # Use first image as sample
        if samp is not None:
            und_s = Calib.undistort(samp, K, D, keep_size=True)  # Remove distortion
            pad = np.ones((samp.shape[0], 10, 3), dtype=np.uint8)*255  # White separator
            und = np.hstack([samp, pad, und_s])  # Concatenate: original | pad | undistorted
    
    return fig, gallery, und  # Return 3D plot, overlay gallery, and undistortion preview

def create_interface():
    """Create and return the Gradio web interface for camera calibration"""
    # Create main interface with custom title and soft theme
    with gr.Blocks(title="Camera Calibration (OpenCV + Gradio)", theme=gr.themes.Soft()) as demo:
        # Add header with usage instructions
        gr.Markdown(
            "## Camera Calibration (OpenCV + Gradio)\n"
            "Upload chessboard images, run calibration, and visualize results.\n"
            "- Images: **/content/images** (Colab) or `./images`\n"
            "- Results: **calibration.json** in project root\n"
            "- Pattern size = INNER corners (e.g., 9×6)\n"
        )
        # File upload section
        with gr.Row():
            # Multi-file uploader restricted to image types, returns file paths
            uploader = gr.Files(label="Upload .jpg/.jpeg/.png images (optional - samples auto-loaded)", file_types=["image"], type="filepath")
            upload_btn = gr.Button("Save to images/")  # Button to process uploaded files
            samples_btn = gr.Button("Load Sample Images", variant="secondary")  # Button to load samples
        # Non-editable text box to show upload status messages  
        upload_msg = gr.Textbox(label="Upload Log", interactive=False)

        # Collapsible section for calibration parameters (starts expanded)
        with gr.Accordion("Calibration Parameters", open=True):
            with gr.Row():
                # Numeric inputs for chessboard pattern (integer values only)
                cols = gr.Number(value=9, label="Inner corners (cols)", precision=0)
                rows = gr.Number(value=6, label="Inner corners (rows)", precision=0)
                # Physical square size in meters (float)
                sq   = gr.Number(value=0.025, label="Square size (meters)")
            with gr.Row():
                # Button to test corner detection without running full calibration
                test_btn = gr.Button("Test detection (no calibration)")
                # Primary button to execute calibration (highlighted)
                run_btn  = gr.Button("Run Calibration", variant="primary")
            # Large text area to display calibration results and status
            calib_out = gr.Textbox(label="Calibration Output", lines=12, interactive=False)

        # Collapsible section for visualization controls (starts expanded)
        with gr.Accordion("Visualizations", open=True):
            with gr.Row():
                # Slider to control number of overlay images shown (3-12)
                max_ov = gr.Slider(3, 12, value=6, step=1, label="Max overlay images")
                # Toggle between interactive (Plotly) and static (Matplotlib) 3D plots
                use_plotly = gr.Checkbox(value=True, label="Interactive 3D (Plotly)")
                # Scale factor for drawn coordinate axes relative to square size
                axis_scale = gr.Slider(1, 10, value=3, step=0.5, label="Axis scale (× square size)")
                # Whether to draw detected chessboard corners on overlay images
                draw_corners = gr.Checkbox(value=True, label="Draw detected corners")
                # Where to place coordinate axes on the chessboard
                anchor = gr.Dropdown(choices=["origin", "center"], value="origin", label="Axes anchor")
            # Plot area for 3D camera pose visualization
            pose_plot = gr.Plot(label="Camera Poses (3D)")
            # Image gallery showing calibration images with projected 3D axes
            gallery   = gr.Gallery(label="Sample images with axes", columns=3, height=300)
            # Single image showing before/after undistortion comparison
            und_img   = gr.Image(label="Undistort Preview (left=original, right=undistorted)")
            # Button to generate all visualizations
            viz_btn   = gr.Button("Generate Visualizations")

        # Connect UI components to their corresponding functions
        upload_btn.click(save_uploads, inputs=[uploader], outputs=[upload_msg])
        samples_btn.click(load_sample_images, outputs=[upload_msg])
        test_btn.click(test_detection, inputs=[cols, rows], outputs=[calib_out])
        run_btn.click(run_calibration, inputs=[cols, rows, sq], outputs=[calib_out])
        viz_btn.click(show_poses_and_overlays,
                      inputs=[max_ov, use_plotly, axis_scale, draw_corners, anchor],
                      outputs=[pose_plot, gallery, und_img])
        
        # Auto-load sample images when interface starts
        demo.load(load_sample_images, outputs=[upload_msg])
        
    return demo  # Return the configured Gradio interface

# Entry point when script is run directly (not imported)
if __name__ == "__main__":
    demo = create_interface()  # Create the Gradio interface
    demo.launch(share=True, debug=True)  # Launch with public sharing and debug info
