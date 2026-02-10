# Watermark Remover for Video (AI-Powered)

This repository provides an AI-powered tool to remove watermarks from videos and images. It leverages Microsoft's **Florence-2** for precise watermark detection and **LaMA (Large Mask Inpainting)** for seamless removal.

## 🚀 Optimized for Google Colab

This version is specifically optimized for **Google Colab**, ensuring:
- **Audio Preservation**: Merges original audio back into the processed video.
- **High Quality**: Uses `libx264` with high-quality settings (`CRF 18`).
- **GPU Acceleration**: Automatically utilizes Colab's T4 GPU for fast processing.

---

## 📖 How to Use on Google Colab

### 1. Set up the Environment
Open a new notebook in [Google Colab](https://colab.research.google.com/) and ensure you are using a **GPU** (Edit -> Notebook settings -> Hardware accelerator -> T4 GPU).

### 2. Prepare the Repository
Run this in a code cell:
```python
!git clone https://github.com/ajithvnr2001/watermark-remover-video.git
%cd watermark-remover-video
!apt-get install -y ffmpeg
```

### 3. Run the Remover
Upload your video/image to the Colab sidebar (folder icon on the left) and run:

```python
# Basic usage for a video
!python colab_remover.py "your_video.mp4" --output "output_folder"

# For high precision (handles fade-in/out and every frame)
!python colab_remover.py "your_video.mp4" --output "output_folder" --skip 1 --fade-in 0.5 --fade-out 0.5
```

---

## 🛠️ Key CLI Options

| Option | Description |
|--------|-------------|
| `--output` | Output directory where processed files are saved. |
| `--skip` | Processes every N frames (1-10). Use `1` for best quality. |
| `--max-bbox` | Max watermark size as % of image (default: 10). Increase if watermark is large. |
| `--fade-in` | Extend mask backwards by N seconds (for fading watermarks). |
| `--fade-out` | Extend mask forwards by N seconds (for fading watermarks). |

---

## 📁 Repository Structure
- `remwm.py`: Core AI logic for detection and inpainting.
- `colab_remover.py`: Wrapper script for dependency installation and high-quality encoding.
- `utils.py`: Support functions for image processing.
- `requirements.txt`: Python dependencies.

## 📜 Credits
Based on the original implementation by [D-Ogi](https://github.com/D-Ogi/WatermarkRemover-AI).
