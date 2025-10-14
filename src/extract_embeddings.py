"""
extract_embeddings.py
---------------------
Uses a trained ResNetAutoencoder to extract latent embeddings (z_audio)
for all Mel spectrograms in data/processed/mel_specs/.

Outputs:
- data/processed/embeddings.csv  (CSV of song_id, segment_id, valence, arousal, z1..zN)
- data/processed/embeddings.pt   (optional PyTorch tensor file)
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

from mel_autoencoder import ResNetAutoencoder
from datasets import MelSpectrogramDataset, pad_or_crop_batch
from config import PROCESSED_DIR, MODEL_DIR, BATCH_SIZE


# ---------------------------------------------------------------------
# Configuration-
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 256
OUTPUT_CSV = PROCESSED_DIR.parent / "embeddings.csv"
OUTPUT_PT = PROCESSED_DIR.parent / "embeddings.pt"
MODEL_PATH = MODEL_DIR / "autoencoder" / "resnet_autoencoder_final.pt"


# Embedding Extraction Function
@torch.no_grad()
def extract_embeddings():
    """
    Loads trained model and extracts latent vectors for all Mel spectrograms.
    """
    print(f"[INFO] Loading model from {MODEL_PATH}")
    model = ResNetAutoencoder(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    print(f"[INFO] Loading dataset from {PROCESSED_DIR}")
    dataset = MelSpectrogramDataset(PROCESSED_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=pad_or_crop_batch)

    all_embeddings = []
    all_metadata = []

    print("[INFO] Extracting embeddings...")
    for X, y in tqdm(loader, total=len(loader)):
        X = X.to(DEVICE)
        _, z = model(X)  # shape: (B, latent_dim)
        all_embeddings.append(z.cpu().numpy())
        all_metadata.append(y.cpu().numpy())

    Z = np.concatenate(all_embeddings, axis=0)
    meta = np.concatenate(all_metadata, axis=0)
    
    z_min, z_max = Z.min(), Z.max()
    Z = (Z - z_min) / (z_max - z_min + 1e-8)

    print(f"[INFO] Saving embeddings... shape={Z.shape}")
    np.save(OUTPUT_PT, Z)
    print(f"[INFO] Saved tensor embeddings: {OUTPUT_PT}")

    df = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])])

    # Try to include song_id, valence, and arousal from mel_metadata.csv
    meta_csv = PROCESSED_DIR.parent / "mel_metadata.csv"
    if meta_csv.exists():
        mel_meta = pd.read_csv(meta_csv)

        # Build a mapping from song_id → valence/arousal
        meta_csv = PROCESSED_DIR.parent / "mel_metadata.csv"
    if meta_csv.exists():
        mel_meta = pd.read_csv(meta_csv)
        mel_meta["song_id"] = mel_meta["song_id"].astype(int)

        # Extract song_id from mel filenames as integer
        song_ids = [int(Path(f).stem.split("_seg")[0]) for f in sorted(PROCESSED_DIR.glob("*.npy"))]

        # Align lengths (truncate or pad if needed)
        n = min(len(song_ids), len(df))
        song_ids = song_ids[:n]
        df = df.iloc[:n].copy()

        df.insert(0, "song_id", song_ids)

        # Merge by song_id — exact match on integer keys
        df = df.merge(mel_meta[["song_id", "valence", "arousal"]], on="song_id", how="left")

        # Fill missing with neutral midpoint
        df.loc[:, "valence"] = df["valence"].fillna(5.0)
        df.loc[:, "arousal"] = df["arousal"].fillna(5.0)


        print(f"[INFO] Successfully merged emotion labels: {df['valence'].isna().sum()} missing")
    else:
        print("[WARN] No mel_metadata.csv found — defaulting val/arousal=5.0.")
        df["song_id"] = range(len(df))
        df["valence"] = 5.0
        df["arousal"] = 5.0

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved embeddings CSV with emotion columns: {OUTPUT_CSV}")
    print("[SUCCESS] Embedding extraction complete.")
    return df


if __name__ == "__main__":
    extract_embeddings()
