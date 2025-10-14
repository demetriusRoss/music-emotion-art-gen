"""
parse_annotations.py
---------------------
Merges DEAM static average annotations (valence + arousal per song) into a single CSV.

Handles folder layout:
data/raw/annotations/annotations averaged per song/song_level/
    ├── static_annotations_averaged_songs_1_2000.csv
    ├── static_annotations_averaged_songs_2000_2058.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
META_DIR = ROOT / "data" / "metadata"


def find_static_avg_files():
    """
    Locate static averaged song annotation files under data/raw.
    """
    files = list(RAW_DIR.rglob("static_annotations_averaged_songs_*.csv"))
    if not files:
        print("[ERROR] Could not find DEAM static averaged files under data/raw.")
        print("[DEBUG] Listing all .csv files found:")
        for f in RAW_DIR.rglob("*.csv"):
            print(" -", f.relative_to(RAW_DIR))
        raise FileNotFoundError("Missing static_annotations_averaged_songs files.")
    print(f"[INFO] Found {len(files)} static annotation files.")
    return sorted(files)


def load_and_merge_annotations():
    """
    Load and concatenate the static averaged song-level DEAM annotation files.
    Each file includes song_id, valence_mean, and arousal_mean columns.
    """
    files = find_static_avg_files()
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "valence_mean" in df.columns and "arousal_mean" in df.columns:
        df = df.rename(columns={"valence_mean": "valence", "arousal_mean": "arousal"})

    if "song_id" not in df.columns:
        df.insert(0, "song_id", range(1, len(df) + 1))

    print(f"[INFO] Combined annotations shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def attach_audio_paths(df):
    """
    Attach audio paths for DEAM songs if available locally.
    """
    audio_dir = RAW_DIR / "MEMD_audio"
    if not audio_dir.exists():
        print("[WARN] Audio directory not found at", audio_dir)
        df["audio_path"] = None
        return df

    audio_files = list(audio_dir.rglob("*.mp3")) + list(audio_dir.rglob("*.wav"))
    print(f"[INFO] Found {len(audio_files)} audio files in DEAM_audio")

    audio_map = {}
    for f in audio_files:
        song_id = f.stem.replace("song_", "")
        audio_map[song_id] = str(f.resolve())

    df["audio_path"] = df["song_id"].astype(str).map(audio_map)
    missing = df["audio_path"].isna().sum()
    if missing:
        print(f"[WARN] Missing audio paths for {missing} songs")
    return df


def save_metadata(df):
    META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = META_DIR / "deam_annotations.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved merged annotations to {out_path}")
    print(df[["valence", "arousal"]].describe().round(2))
    return out_path


def generate_deam_metadata():
    """
    Full pipeline: load, merge, attach audio paths, and save CSV.
    """
    df = load_and_merge_annotations()
    df = attach_audio_paths(df)
    csv_path = save_metadata(df)
    return csv_path


if __name__ == "__main__":
    generate_deam_metadata()
