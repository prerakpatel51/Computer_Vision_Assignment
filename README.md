# Camera Calibration (OpenCV + Gradio)

This project calibrates your camera from chessboard images and provides an **inline Gradio UI** (Colab-friendly) to upload images, run calibration, view **camera poses**, inspect **axes overlays**, and preview **undistortion**.

## 📦 Files
- `calibration_utils.py` – core, stateless helpers (IO, Board, Img, Calib, Overlay, Render).
- `app_gradio.py` – Gradio interface using the helpers.
- `Camera_Calibration.ipynb` – Notebook (Colab/local) that launches the UI inline.
- `requirements.txt` – Python dependencies.
- `ai_tools_appendix_template.md` – Template for your AI usage appendix.

## 🚀 Colab / Notebook Usage
1. Open `Camera_Calibration.ipynb` in Google Colab.
2. Run the first cell to install packages.
3. Run the cell that creates/loads the Gradio UI. It will appear **inline**.
4. **Upload** your `.jpg/.jpeg/.png` chessboard images (≥15 recommended) via the UI.
5. Set the chessboard **inner corners** (default `9×6`) and **square size** (e.g., `0.025 m`).
6. Click **Run Calibration**.
7. Click **Generate Visualizations** to see:
   - 3D camera poses (interactive Plotly),
   - 5–10 images with axes overlays,
   - side-by-side **undistort** preview.

> Chessboard: https://docs.opencv.org/4.x/pattern.png

## 🚀 Google Colab Usage
```bash
!git clone https://github.com/prerakpatel51/Computer_Vision_Assignment.git
%cd Computer_Vision_Assignment/
from camera_calibration import run_calibration_pipeline
run_calibration_pipeline()
```

## 📓 Jupyter Notebook (Local) Usage
```bash
!git clone https://github.com/prerakpatel51/Computer_Vision_Assignment.git
%cd Computer_Vision_Assignment/
from camera_calibration import run_calibration_pipeline
run_calibration_pipeline()
```

## 🧪 Local (Desktop/Terminal) Usage

**Option 1: Using the consolidated script (recommended)**
```bash
git clone https://github.com/prerakpatel51/Computer_Vision_Assignment.git
cd Computer_Vision_Assignment/
python camera_calibration.py
# This will auto-install dependencies and launch the Gradio interface
```

**Option 2: Manual setup**
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app_gradio.py
# open http://127.0.0.1:7860
```

## 📄 Output
- Calibration saved as `calibration.json`, including:
  - `K` (intrinsic matrix), `D` (distortion),
  - `image_size`, `pattern_size`, `square_size_m`,
  - `num_views`, `per_view_errors`,
  - `extrinsics` (R,t) per image,
  - `valid_paths` for reproducibility.

## 🧠 Notes
- Use sharp, well‑lit images and vary distance/tilt/roll.
- Pattern size is **inner corners** (squares minus one).
- If results look off, drop outlier views (highest per‑view error) and recalibrate.
