"""
datasets.py
------------
Dataset class for loading precomputed Mel spectrogram .npy files.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch.nn.functional as F


class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, metadata_csv=None, normalize=True):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.data_dir}")

        self.normalize = normalize

        # optional metadata
        if metadata_csv is None:
            metadata_csv = self.data_dir.parent / "mel_metadata.csv"
        self.meta = pd.read_csv(metadata_csv) if metadata_csv.exists() else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        mel = np.load(file_path)

        # ensure 2D float32 array
        mel = mel.astype(np.float32)
        if self.normalize:
            mel_min, mel_max = mel.min(), mel.max()
            if mel_max > mel_min:  # prevent divide-by-zero
                mel = (mel - mel_min) / (mel_max - mel_min)

        # add channel dimension -> (1, n_mels, time)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)

        # optional regression targets
        if self.meta is not None:
            song_id = file_path.stem.split("_")[0]
            row = self.meta[self.meta["song_id"] == song_id]
            if not row.empty:
                valence = float(row.iloc[0].get("valence", 0.0))
                arousal = float(row.iloc[0].get("arousal", 0.0))
                target = torch.tensor([valence, arousal], dtype=torch.float32)
            else:
                target = torch.zeros(2, dtype=torch.float32)
        else:
            target = torch.zeros(2, dtype=torch.float32)

        return mel_tensor, target


def pad_or_crop_batch(batch, target_len=645):
    """
    Pads or crops each Mel spectrogram in the batch to have the same time dimension.
    """
    mels, targets = zip(*batch)
    padded = []

    for mel in mels:
        _, n_mels, time_len = mel.shape
        if time_len < target_len:
            pad_amt = target_len - time_len
            mel = F.pad(mel, (0, pad_amt))
        elif time_len > target_len:
            mel = mel[:, :, :target_len]
        padded.append(mel)

    X = torch.stack(padded)
    Y = torch.stack(targets)
    return X, Y