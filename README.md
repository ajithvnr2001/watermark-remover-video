# AI Watermark Remover for Video (Colab Ready)

The simplest AI tool to remove watermarks from videos and images. Optimized for Google Colab with high-quality output and audio preservation.

## ✨ Features
- **One-Command Setup**: Automatically installs all AI models and dependencies.
- **Folder Support**: Pass a folder path to process all videos/images at once.
- **Smart Naming**: Saves results as `filename_wmr.mp4` in the same folder.
- **High Quality**: Uses LaMA inpainting and high-quality `libx264` encoding.
- **Audio Preservation**: Keeps the original audio from your videos.

---

## 🚀 How to Use on Google Colab (Fastest)

### 1. Set Up Google Colab
Open [Google Colab](https://colab.research.google.com/) and go to **Edit -> Notebook settings**. Select **T4 GPU** as the hardware accelerator.

### 2. Run the Program
Copy and paste this into a code cell in Colab:

```python
# 1. Clone the repository
!git clone https://github.com/ajithvnr2001/watermark-remover-video.git
%cd watermark-remover-video
!apt-get install -y ffmpeg

# 2. Run on a folder (Upload your videos/images to a folder in Colab first)
!python colab_remover.py "/content/your_folder"
```

---

## 🛠️ Advanced Options

You can adjust the processing by adding these flags:

```bash
# Example with options:
!python colab_remover.py "/content/input" --skip 1 --fade-in 0.5 --max-bbox 15
```

| Flag | Description |
|------|-------------|
| `--skip` | Processes every N frames (1-10). Default is `1` (best quality). |
| `--max-bbox` | Max watermark size as % of image. Default is `10`. Increase if watermark is huge. |
| `--fade-in` | Seconds to look *before* a watermark appears (for fading watermarks). |
| `--fade-out` | Seconds to keep masking *after* a watermark disappears. |

---

## 📁 Repository Structure
- `colab_remover.py`: The only file you need to run. Handles everything automatically.
- `remwm.py`: The engine for AI detection and removal.
- `utils.py`: Support utilities.
- `requirements.txt`: List of dependencies.

## 📜 Credits
Based on the original implementation by [D-Ogi](https://github.com/D-Ogi/WatermarkRemover-AI).
