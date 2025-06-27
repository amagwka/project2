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
