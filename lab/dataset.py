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
    """Dataset for sequences of cached DINOv2 embeddings stored in an H5 file."""

    def __init__(self, h5_path: str, sequence_length: int = 30, frame_gap: int = 1,
                 in_memory: bool = True) -> None:
        """Parameters
        ----------
        h5_path: str
            Path to the H5 file containing embeddings.
        sequence_length: int
            Number of consecutive embeddings that form the input sequence.
        frame_gap: int
            Target frames ahead to predict.
        in_memory: bool
            When True (default) the entire H5 file is loaded into RAM on
            construction which allows using multiple dataloader workers
            without each worker reopening the file. This greatly speeds up
            training for small and medium sized datasets.
        """

        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        self.in_memory = in_memory

        import h5py
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())
            self.lengths = {k: f[k].shape[0] for k in self.keys}
            self.cache = None
            if in_memory:
                self.cache = {k: torch.tensor(f[k][:], dtype=torch.float32)
                              for k in self.keys}

        self.indices = []
        for k, length in self.lengths.items():
            max_start = length - sequence_length - frame_gap
            for i in range(max_start + 1):
                self.indices.append((k, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key, start = self.indices[idx]

        if self.cache is not None:
            data_seq = self.cache[key]
            data = data_seq[start:start + self.sequence_length]
            target = data_seq[start + self.sequence_length + self.frame_gap - 1]
        else:
            import h5py
            with h5py.File(self.h5_path, "r") as f:
                data = torch.tensor(f[key][start:start + self.sequence_length],
                                   dtype=torch.float32)
                target = torch.tensor(
                    f[key][start + self.sequence_length + self.frame_gap - 1],
                    dtype=torch.float32)

        return data, target

