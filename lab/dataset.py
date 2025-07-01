import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision import transforms

def _default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class EmbeddingH5Dataset(Dataset):
    def __init__(self, h5_path: str, sequence_length: int = 30, frame_gap: int = 1):
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
        return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

