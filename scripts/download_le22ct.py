"""
Download LE22ct Dataset
=======================
Downloads the exact dataset used by CT-EFT-20 for fair comparison.

Dataset: LE22ct (Lichess Elite 2022, checkmate, time-controlled)
Source: Lichess Elite Database (2400+ vs 2200+ games)
Size: 13,287,522 positions
Format: H5 file
URL: https://chesstransformers.blob.core.windows.net/data/LE22ct.zip
"""

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys


def download_file(url: str, destination: Path) -> bool:
    """Download file with progress bar."""
    print(f"\n📥 Downloading from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=destination.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file."""
    print(f"\n📦 Extracting {zip_path.name}...")
    
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            print(f"   Files in archive: {len(file_list)}")
            
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, extract_to)
        
        print(f"✅ Extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def main():
    print("=" * 70)
    print("LE22ct Dataset Downloader")
    print("(CT-EFT-20's Training Dataset)")
    print("=" * 70)
    print()
    print("This dataset contains:")
    print("  - 13,287,522 chess positions")
    print("  - From Lichess Elite games (2400+ vs 2200+)")
    print("  - Checkmate games only")
    print("  - Time control ≥ 5 minutes")
    print("  - Winning moves labeled")
    print()
    
    # URLs
    url = "https://chesstransformers.blob.core.windows.net/data/LE22ct.zip"
    
    # Paths
    raw_dir = Path("dataset/raw")
    zip_path = raw_dir / "LE22ct.zip"
    extract_dir = raw_dir / "LE22ct"
    
    # Download
    if zip_path.exists():
        print(f"✓ {zip_path} already exists")
    else:
        if not download_file(url, zip_path):
            print("\n❌ Download failed!")
            sys.exit(1)
    
    # Extract
    if extract_dir.exists() and list(extract_dir.glob('*.h5')):
        print(f"✓ {extract_dir} already extracted")
    else:
        if not extract_zip(zip_path, extract_dir):
            print("\n❌ Extraction failed!")
            sys.exit(1)
    
    # Check what we got
    h5_files = list(extract_dir.glob('*.h5'))
    
    print("\n" + "=" * 70)
    print("✅ Dataset Download Complete!")
    print("=" * 70)
    print(f"\nH5 files found: {len(h5_files)}")
    for h5_file in h5_files:
        size_mb = h5_file.stat().st_size / 1024 / 1024
        print(f"  - {h5_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n📋 Next step:")
    print(f"   python scripts/preprocess_le22ct.py")
    print()


if __name__ == "__main__":
    main()

