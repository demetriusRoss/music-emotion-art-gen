import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch
from abstract_renderer import render_abstract
from torchvision import transforms

class EmbeddingsToArtDataset(Dataset):
    """
    Loads rows from data/processed/embeddings.csv and for each:
      - condition vector: concat(z_audio (256D), [valence, arousal])
      - target image: synthetic abstract image generated from val/arousal (+ seed)
    """

    def __init__(
        self,
        embeddings_csv,
        img_size=128,
        use_valence_arousal=True,
        seed_base=1234,
        normalize=True,
        augment=True,
    ):
        self.emb = pd.read_csv(embeddings_csv)
        self.img_size = img_size
        self.use_va = use_valence_arousal
        self.seed_base = seed_base

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.35,
                    hue=0.05
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.z_cols = [c for c in self.emb.columns if c.startswith("z")]
        assert len(self.z_cols) > 0, "No z* columns found in embeddings CSV."

        # Fill in any missing values for valence/arousal
        self.emb["valence"] = self.emb["valence"].fillna(5.0)
        self.emb["arousal"] = self.emb["arousal"].fillna(5.0)

        # Normalize embeddings if needed
        if normalize:
            Z = self.emb[self.z_cols].values.astype(np.float32)
            mean, std, maxv = Z.mean(), Z.std(), Z.max()

            if std < 0.5 or maxv < 0.8:
                print(f"[WARN] Embeddings look compressed (mean={mean:.3f}, std={std:.3f}, max={maxv:.3f}) → applying z-score normalization.")
                Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)
                self.emb[self.z_cols] = Z
                print(f"[INFO] Normalized embeddings: mean={Z.mean():.4f}, std={Z.std():.4f}")
            else:
                print("[INFO] Embeddings appear normalized — skipping re-normalization.")

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        row = self.emb.iloc[idx]
        z = row[self.z_cols].to_numpy(dtype=np.float32)

        v = np.clip(float(row["valence"]) / 5.0, 0.0, 1.0)
        a = np.clip(float(row["arousal"]) / 5.0, 0.0, 1.0)

        cond = np.concatenate([z, np.array([v, a], dtype=np.float32)], axis=0) if self.use_va else z

        seed = self.seed_base + idx
        img = render_abstract(v * 10.0, a * 10.0, size=self.img_size, seed=seed)  # returns HxWx3 (float 0–1 or 0–255)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)  # ensure uint8 for PIL
        img = self.transform(img)                          # apply augmentations Tensor [0,1]
        
        return (
            img,                                          # target image
            torch.from_numpy(cond).float(),               # condition vector
            torch.tensor([v, a], dtype=torch.float32),    # valence/arousal for plotting
        )
