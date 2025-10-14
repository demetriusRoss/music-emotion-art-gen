import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "mel_specs"
META_DIR = DATA_DIR / "metadata"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "experiments" / "logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 22050
N_MELS = 128
SEGMENT_DURATION = 10

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 256

SEED = 42
torch.manual_seed(SEED)
