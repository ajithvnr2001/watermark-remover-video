import os
import sys
import subprocess
from pathlib import Path

# Move external imports inside to avoid ModuleNotFoundError before installation
# from loguru import logger
# import torch

# 1. Install dependencies if not present
def install_dependencies():
    print("Checking and installing dependencies...")
    try:
        import iopaint
        import transformers
        print("Dependencies already installed.")
    except ImportError:
        print("Installing dependencies... This may take a minute.")
        # Install core dependencies with specific versions to avoid conflicts
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                        "transformers>=4.50.0", "diffusers>=0.30.0", "numpy<2",
                        "opencv-python-headless", "Pillow>=10.0.0", "loguru", "click", "tqdm", "psutil", "pyyaml"])
        
        # Install iopaint without dependencies to avoid resolver conflicts
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "iopaint"])
        
        # Install iopaint's required sub-dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                        "pydantic", "typer", "einops", "omegaconf", "easydict", "yacs"])
        
        print("Dependencies installed successfully.")

# 2. Setup models
def setup_models():
    print("Setting up models...")
    # Florence-2 will be downloaded by transformers on first use
    # LaMA needs to be downloaded to torch hub
    lama_dir = Path.home() / ".cache/torch/hub/checkpoints"
    lama_file = lama_dir / "big-lama.pt"
    
    if not lama_file.exists():
        print("Downloading LaMA model (~196MB)...")
        lama_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
        urllib.request.urlretrieve(url, lama_file)
        print("LaMA model downloaded.")
    else:
        print("LaMA model already exists.")

# 2.5 Patch remwm.py for quality
def patch_remwm():
    print("Patching remwm.py for better quality...")
    remwm_path = Path("remwm.py")
    if not remwm_path.exists():
        print("remwm.py not found, skipping patch.")
        return
    
    with open(remwm_path, "r") as f:
        content = f.read()
    
    # Update FFmpeg merge command to use libx264 instead of copy
    # This ensures high quality re-encoding
    old_code = '"-c:v", "copy",'
    new_code = '"-c:v", "libx264", "-crf", "18", "-preset", "veryfast",'
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(remwm_path, "w") as f:
            f.write(content)
        print("remwm.py patched successfully (High Quality enabled).")
    else:
        print("remwm.py already patched or code structure changed.")

# 3. Main processing function
def process_watermark(input_path, output_dir, detection_skip=1, fade_in=0.0, fade_out=0.0, max_bbox=10.0):
    # Import locally after installation
    import torch
    from loguru import logger
    from remwm import main as remwm_main
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for the CLI
    args = [
        str(input_path),
        str(output_dir),
        "--max-bbox-percent", str(max_bbox),
        "--detection-skip", str(detection_skip),
        "--fade-in", str(fade_in),
        "--fade-out", str(fade_out),
        "--overwrite"
    ]
    
    print(f"Starting processing: {input_path}")
    print(f"Settings: skip={detection_skip}, fade_in={fade_in}, fade_out={fade_out}, max_bbox={max_bbox}%")
    
    try:
        # We use sys.argv trick to call the click-managed main function
        sys.argv = ["remwm.py"] + args
        remwm_main()
        print(f"Processing complete! Results saved in: {output_dir}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watermark Removal for Google Colab")
    parser.add_argument("input", help="Path to input image, video, or folder")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--skip", type=int, default=1, help="Detection skip for videos (1-10, default: 1)")
    parser.add_argument("--fade-in", type=float, default=0.0, help="Fade-in expansion in seconds")
    parser.add_argument("--fade-out", type=float, default=0.0, help="Fade-out expansion in seconds")
    parser.add_argument("--max-bbox", type=float, default=10.0, help="Max watermark size as % of image (default: 10)")
    parser.add_argument("--setup", action="store_true", help="Only run setup (install dependencies and models)")

    args = parser.parse_args()

    # Run setup
    install_dependencies()
    setup_models()
    patch_remwm()
    
    if not args.setup:
        process_watermark(args.input, args.output, args.skip, args.fade_in, args.fade_out, args.max_bbox)
