"""
audio_preprocessing.py
----------------------
Extracts Mel spectrograms from MEMD/DEAM audio files
and saves them as .npy arrays in data/processed/mel_specs.

Skips reprocessing if spectrograms already exist.
"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"
PROC_DIR = ROOT / "data" / "processed"
AUDIO_DIR = RAW_DIR / "MEMD_audio"

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
SEGMENT_DURATION = 15  
TARGET_DIR = PROC_DIR / "mel_specs"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_metadata():
    csv_path = META_DIR / "deam_annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded metadata: {df.shape} entries")
    return df


def extract_mel_spectrogram(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except Exception as e:
        print(f"[ERROR] Could not process {audio_path}: {e}")
        return None


def segment_and_save_mel(mel, song_id, output_dir=TARGET_DIR, segment_seconds=SEGMENT_DURATION):
    sr_per_hop = HOP_LENGTH / SAMPLE_RATE
    frames_per_segment = int(segment_seconds / sr_per_hop)
    segments = mel.shape[1] // frames_per_segment

    saved_files = []
    for i in range(segments):
        seg = mel[:, i * frames_per_segment:(i + 1) * frames_per_segment]
        if seg.shape[1] < frames_per_segment // 2:
            continue
        fname = f"{song_id}_seg{i+1}.npy"
        out_path = output_dir / fname
        np.save(out_path, seg)
        saved_files.append(out_path)
    return saved_files


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def preprocess_audio(skip_existing=True):
    """
    Main pipeline:
    - Loads metadata
    - Checks for existing Mel files
    - Extracts and saves Mel spectrograms only for missing audio
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()

    existing_mels = list(TARGET_DIR.glob("*.npy"))
    existing_ids = {f.name.split("_")[0] for f in existing_mels}
    print(f"[INFO] Found {len(existing_mels)} existing Mel files ({len(existing_ids)} songs).")

    total_songs = len(df)
    if skip_existing and len(existing_ids) / total_songs >= 0.9:
        print(f"[INFO] {len(existing_ids)} / {total_songs} songs already processed (>90%). Skipping extraction.")
        return

    processed_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        song_id = str(row["song_id"])
        if skip_existing and song_id in existing_ids:
            continue  # skip already processed songs

        # Locate audio
        audio_path = None
        for ext in [".mp3", ".wav"]:
            candidate = AUDIO_DIR / f"{song_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            continue

        mel = extract_mel_spectrogram(audio_path)
        if mel is None:
            continue

        saved_files = segment_and_save_mel(mel, song_id)
        for f in saved_files:
            processed_records.append({
                "song_id": song_id,
                "mel_path": str(f.relative_to(ROOT)),
                "valence": row.get("valence"),
                "arousal": row.get("arousal")
            })

    # Merge with existing metadata if already present
    mel_meta_path = PROC_DIR / "mel_metadata.csv"
    if mel_meta_path.exists():
        existing_df = pd.read_csv(mel_meta_path)
        mel_meta_df = pd.concat([existing_df, pd.DataFrame(processed_records)], ignore_index=True)
        mel_meta_df = mel_meta_df.drop_duplicates(subset=["mel_path"])
    else:
        mel_meta_df = pd.DataFrame(processed_records)

    mel_meta_df.to_csv(mel_meta_path, index=False)
    print(f"[INFO] Saved Mel spectrogram metadata: {len(mel_meta_df)} entries total.")


if __name__ == "__main__":
    preprocess_audio(skip_existing=True)
