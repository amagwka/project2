import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import transforms

try:
    from dinov2.models import vit_small
    from dinov2.models import load_pretrained_weights
except Exception:  # fallback if dinov2 is not installed
    vit_small = None


def _default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class VideoDataset(Dataset):
    """Dataset that yields sequences of DINOv2 embeddings from video files."""

    def __init__(self, video_dir: str, sequence_length: int = 16, transform=None):
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length
        self.transform = transform or _default_transform()
        self.video_files = sorted([p for p in self.video_dir.glob('*.mp4')])

        if vit_small is not None:
            self.encoder = vit_small()
            load_pretrained_weights(self.encoder, model_name="dinov2_vits14")
            self.encoder.eval()
        else:
            self.encoder = None

    def __len__(self) -> int:
        return len(self.video_files)

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return torch.randn(frames.shape[0], 384)
        frames = self.transform(frames).unsqueeze(0)
        emb = self.encoder(frames)
        return emb.squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = self.video_files[idx]
        video, _, _ = read_video(str(video_path), pts_unit="sec")
        total_frames = video.shape[0]
        if total_frames <= self.sequence_length:
            start = 0
        else:
            start = torch.randint(0, total_frames - self.sequence_length, (1,)).item()
        seq = video[start:start + self.sequence_length + 1]
        emb_seq = []
        for frame in seq:
            emb_seq.append(self._encode_frames(frame))
        emb_seq = torch.stack(emb_seq)
        return emb_seq[:-1], emb_seq[-1]

class EmbeddingH5Dataset(Dataset):
    """Dataset that yields sequences of embeddings from an H5 file."""

    def __init__(self, h5_path: str, sequence_length: int = 30, frame_gap: int = 30):
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        import h5py
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())
            self.lengths = {k: f[k].shape[0] for k in self.keys}
        self.indices = []
        for k, length in self.lengths.items():
            max_start = length - sequence_length - frame_gap
            for i in range(max_start + 1):
                self.indices.append((k, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import h5py
        key, start = self.indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            data = f[key][start:start + self.sequence_length]
            target = f[key][start + self.sequence_length + self.frame_gap - 1]
        return torch.tensor(data), torch.tensor(target)

