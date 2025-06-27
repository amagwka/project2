# TMDN Lab

This directory contains an implementation of the **Temporal Manifold Diffusion Network (TMDN)** for next-frame prediction using DINOv2 embeddings.

Files:
- `tmdn.py` – the model architecture and helper functions.
- `dataset.py` – simple dataset class that loads video files and generates sequences of DINOv2 embeddings.
- `train_tmdn.py` – example training script for running TMDN on a directory of videos.

## Usage

Install dependencies (requires PyTorch, torchvision and optionally `dinov2`):
```bash
pip install torch torchvision dinov2
```

Run training:
```bash
python train_tmdn.py --data-dir /path/to/videos --epochs 5 --batch-size 4
```

The script will save the trained model weights to `tmdn_model.pt`.

## Image Pipeline

Three helper scripts are provided for workflows that operate on directories of videos and frames:

1. `scripts/mp4_to_png.py` – converts every `.mp4` in a directory into a folder of PNG frames using ffmpeg. Frames are skipped according to a desired FPS without interpolation.
2. `scripts/cache_embeddings.py` – encodes the extracted PNG frames with DINOv2 small and stores the embeddings in an H5 file.
3. `scripts/train_from_h5.py` – trains the TMDN model using the cached embeddings.

The default dataset settings use a history of 30 frame embeddings and predict one frame 30 frames later (i.e. 1 second at 30 fps).

Example usage:
```bash
# 1. Extract frames
python scripts/mp4_to_png.py --video-dir /path/to/videos --out-dir frames --fps 1

# 2. Cache embeddings
python scripts/cache_embeddings.py --image-dir frames --output data.h5

# 3. Train model
python scripts/train_from_h5.py --h5 data.h5 --epochs 5 --batch-size 4
```
