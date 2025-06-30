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

1. `scripts/mp4_to_png.py` – converts each `.mp4` into a folder of 224×224 PNG frames using ffmpeg. Frames are skipped according to a desired FPS without interpolation.
2. `scripts/cache_embeddings.py` – encodes the extracted PNG frames with Hugging Face DINO models (`small`, `large` or `aimv2`) and stores the embeddings in an H5 file.

3. `scripts/train_from_h5.py` – trains the TMDN model using the cached embeddings.

The default dataset settings use a history of 30 frame embeddings and predict the next frame by default. Pass ``--frame-gap`` to train the models to predict further into the future.

Example usage:
```bash
# 1. Extract frames
python scripts/mp4_to_png.py --video-dir /path/to/videos --out-dir frames --fps 1

# 2. Cache embeddings (choose model with --model if desired)
python scripts/cache_embeddings.py --image-dir frames --output data.h5 --model small


# 3. Train model
python scripts/train_from_h5.py --h5 data.h5 --epochs 5 --batch-size 4
```

## Experiment 2: LSTM Baseline

A second experiment explores a simpler recurrent approach for next-frame prediction. The model in `rnn_baseline.py` uses an LSTM (or GRU) over the last 30 frame embeddings. With the default dataset at one frame per second this provides a 30‑second history window.

Training from cached embeddings can be done with:
```bash
python scripts/train_rnn.py --h5 data.h5 --model lstm --epochs 5 --batch-size 4
# predict five steps ahead
python scripts/train_rnn.py --h5 data.h5 --frame-gap 5 --epochs 5
```

A larger GRU variant may be trained by increasing the hidden dimension:
```bash
python scripts/train_rnn.py --h5 data.h5 --model gru --hidden-dim 1024
```

In our tests the trained TMDN outperformed the GRU baseline in prediction accuracy even when the GRU used the bigger hidden size.
