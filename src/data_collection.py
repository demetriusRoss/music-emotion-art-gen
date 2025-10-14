import os
import tarfile
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# You already have annotation link via Zenodo (DEAM_Annotations.zip)
DEAM_ANNOT_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip"

# Hypothetical CVML-hosted audio URL (you’ll need to replace with actual)
DEAM_AUDIO_CVML_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip"

def _download_file(url: str, output_path: Path):
    """
    Download a file with a progress bar and content validation.
    """
    print(f"[INFO] Starting download: {url}")
    with requests.get(url, stream=True, allow_redirects=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch {url} (status {r.status_code})")

        # Detect HTML response (e.g., bad redirect)
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(f"Expected binary file, got HTML instead from {url}")

        total = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as fp, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {output_path.name}",
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
                    bar.update(len(chunk))
    print(f"[INFO] Download completed: {output_path}")


def _extract_archive(file_path: Path, extract_to: Path):
    """
    Extract a ZIP or TAR.GZ archive.
    """
    print(f"[INFO] Extracting {file_path.name}...")
    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(extract_to)
    elif file_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(extract_to)
    else:
        print(f"[WARN] Unknown archive format: {file_path}")
    print(f"[INFO] Extraction complete: {extract_to}")
    
    
def fetch_deam_annotations(force: bool = False):
    """
    Download and extract the DEAM Annotations dataset from Zenodo.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    annot_archive = RAW_DIR / "DEAM_Annotations.zip"
    extracted_dir = RAW_DIR / "annotations"

    if extracted_dir.exists() and not force:
        print("[INFO] DEAM annotations already exist at", extracted_dir)
        return extracted_dir

    try:
        _download_file(DEAM_ANNOT_URL, annot_archive)
        _extract_archive(annot_archive, RAW_DIR)
        print("[INFO] DEAM annotations successfully extracted.")
    except Exception as e:
        print("[ERROR] Failed to download annotations:", e)
        print("Please manually download from:")
        print("  https://zenodo.org/records/11400122/files/DEAM_Annotations.zip")
        print(f"Then place it inside: {RAW_DIR}")
    return extracted_dir


def fetch_deam_audio(force: bool = False):
    """
    Download and extract the DEAM raw audio dataset from the official CVML site.
    NOTE: You must verify that the URL in DEAM_AUDIO_CVML_URL is correct.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    audio_archive = RAW_DIR / "DEAM_audio.zip"
    extracted_dir = RAW_DIR / "MEMD_audio"

    if extracted_dir.exists() and not force:
        print("[INFO] DEAM audio already exists at", extracted_dir)
        return extracted_dir

    try:
        print("[INFO] Downloading DEAM raw audio (this may take a while)...")
        _download_file(DEAM_AUDIO_CVML_URL, audio_archive)
        _extract_archive(audio_archive, RAW_DIR)
        print("[INFO] DEAM audio successfully extracted.")
    except Exception as e:
        print("[ERROR] Failed to download audio:", e)
        print("Please manually download the audio archive from:")
        print("  https://cvml.unige.ch/databases/DEAM/")
        print(f"Then place it inside: {RAW_DIR}")
    return extracted_dir

def check_data_status():
    """
    Summarize the contents of data/raw.
    """
    if not RAW_DIR.exists():
        print("[INFO] data/raw directory does not exist yet.")
        return

    files = list(RAW_DIR.glob("**/*"))
    if not files:
        print("[INFO] No files found in data/raw yet.")
        return

    print(f"[INFO] Found {len(files)} files in {RAW_DIR}")
    for f in files[:15]:
        print("  -", f.relative_to(RAW_DIR))
    if len(files) > 15:
        print(f"  ... (+{len(files) - 15} more files)")
