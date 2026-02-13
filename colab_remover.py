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
                        "opencv-python-headless<4.11", "Pillow>=10.0.0", "loguru", "click", "tqdm", "psutil", "pyyaml"])
        
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

# 2.5 Patch remwm.py for quality and bug fixes
def patch_remwm():
    print("Patching remwm.py for quality and stability...")
    remwm_path = Path("remwm.py")
    if not remwm_path.exists():
        print("remwm.py not found, skipping patch.")
        return
    
    with open(remwm_path, "r") as f:
        content = f.read()
    
    # fix 1: Update FFmpeg merge command to use libx264 
    content = content.replace('"-c:v", "copy",', '"-c:v", "libx264", "-crf", "18", "-preset", "veryfast",')
    
    # fix 2: Fix dtype mismatch error (Input type float and bias type Half)
    # Force float32 for Florence-2 loading
    content = content.replace(
        'from_pretrained("florence-community/Florence-2-large").to(device)',
        'from_pretrained("florence-community/Florence-2-large", torch_dtype=torch.float32).to(device)'
    )
    
    # fix 3: Force input dtype match in identity function
    content = content.replace(
        'inputs = {k: v.to(device) for k, v in inputs.items()}',
        'inputs = {k: v.to(device, dtype=model.dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}'
    )

    with open(remwm_path, "w") as f:
        f.write(content)
    print("remwm.py patched successfully.")

# 3. Process a single file
def process_single_file(file_path, detection_skip=1, fade_in=0.0, fade_out=0.0, max_bbox=10.0):
    import torch
    from loguru import logger
    from remwm import main as remwm_main
    
    file_path = Path(file_path)
    if "_wmr" in file_path.stem:
        return

    output_path = file_path.parent / f"{file_path.stem}_wmr{file_path.suffix}"
    
    print(f"\n--- Processing: {file_path.name} ---")
    
    # Prepare arguments for the CLI
    args = [
        str(file_path),
        str(output_path),
        "--max-bbox-percent", str(max_bbox),
        "--detection-skip", str(detection_skip),
        "--fade-in", str(fade_in),
        "--fade-out", str(fade_out),
        "--overwrite"
    ]
    
    try:
        original_argv = sys.argv
        sys.argv = ["remwm.py"] + args
        remwm_main()
        sys.argv = original_argv
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

# 4. Main processing function (handles folder or file)
def process_watermark(input_path, detection_skip=1, fade_in=0.0, fade_out=0.0, max_bbox=10.0):
    import torch
    from loguru import logger
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Only video extensions
        extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
        files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in extensions])
        
        if not files:
            print(f"No video files found in {input_path}")
            return
            
        print(f"Found {len(files)} videos to process in {input_path}")
        for f in files:
            process_single_file(f, detection_skip, fade_in, fade_out, max_bbox)
    else:
        process_single_file(input_path, detection_skip, fade_in, fade_out, max_bbox)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watermark Removal for Google Colab/Local")
    parser.add_argument("input", help="Path to input image, video, or folder")
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
        process_watermark(args.input, args.skip, args.fade_in, args.fade_out, args.max_bbox)
