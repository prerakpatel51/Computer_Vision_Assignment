#!/usr/bin/env python3
"""
Single function to run the entire calibration pipeline.
Usage: python camera_calibration.py or import and call run_calibration_pipeline()
"""

import sys
import os
import subprocess

def run_calibration_pipeline():
    """
    Consolidated function that runs the entire calibration.ipynb pipeline.
    This includes dependency installation, environment setup, and launching the Gradio interface.
    """
    
    print("🚀 Starting Camera Calibration Pipeline...")
    
    # Step 1: Environment Detection
    IN_COLAB = 'google.colab' in sys.modules
    IN_JUPYTER = 'ipykernel' in sys.modules and not IN_COLAB
    print(f"Environment: {'Google Colab' if IN_COLAB else 'Jupyter Notebook' if IN_JUPYTER else 'Script/Terminal'}")
    
    # Step 2: Handle Google Colab setup if needed
    if IN_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            os.chdir('/content/drive/MyDrive/Colab Notebooks/pinhole')
            print("Google Drive mounted and directory changed")
        except Exception as e:
            print(f"⚠️ Colab setup failed: {e}")
    else:
        print(f"Working directory: {os.getcwd()}")
    
    # Step 3: Install Dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Dependency installation failed: {e}")
        return False
    
    # Step 4: Verify critical imports
    try:
        import gradio as gr
        print(f"✅ Gradio verified: {gr.__version__}")
    except ImportError:
        print("⚠️ Critical imports failed. Please restart and try again.")
        return False
    
    # Step 5: Add current directory to path and import calibration utilities
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    
    try:
        from calibration_utils import IO, Calib, Overlay, Render, Img, Board
        print("✅ All calibration functions imported successfully")
        print("Available classes: IO, Calib, Overlay, Render, Img, Board")
    except ImportError as e:
        print(f"⚠️ Failed to import calibration utilities: {e}")
        return False
    
    # Step 6: Launch Gradio Interface
    try:
        # Force reload modules in case they're cached (important for Colab)
        import importlib
        if 'app_gradio' in sys.modules:
            importlib.reload(sys.modules['app_gradio'])
        
        from app_gradio import create_interface, load_default_samples
        demo = create_interface()
        
        # Test that the load_default_samples function exists
        print("✅ Load default samples function available:", hasattr(load_default_samples, '__call__'))
        
        print("🚀 Launching Gradio interface...")
        if IN_COLAB:
            demo.launch(share=True, inline=True, debug=True)
        elif IN_JUPYTER:
            demo.launch(share=False, inline=True, debug=False)
        else:
            # For script execution, launch with share=False and don't block
            demo.launch(share=False, debug=False)
            
        print("✅ Calibration pipeline launched successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to launch Gradio interface: {e}")
        return False

if __name__ == "__main__":
    success = run_calibration_pipeline()
    if not success:
        sys.exit(1)