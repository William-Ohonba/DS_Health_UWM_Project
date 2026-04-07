import os
import subprocess
from pathlib import Path
import glob

DATA_DIR = Path("data")

def download_data():
    if (DATA_DIR / "train.csv").exists():
        print(" Data already exists, skipping download.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading dataset from Kaggle...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "uw-madison-gi-tract-image-segmentation",
        "-p", str(DATA_DIR)
    ], check=True)

    zips = glob.glob(str(DATA_DIR / "*.zip"))
    for zpath in zips:
        subprocess.run(["unzip", "-o", zpath, "-d", str(DATA_DIR)], check=True)

    print("Done.")

if __name__ == "__main__":
    download_data()