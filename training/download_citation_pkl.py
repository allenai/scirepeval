#!/usr/bin/env python3
"""Download citation pkl files from S3 to local disk for faster testing."""
import os
import sys

print("Downloading citation pkl files from S3...")
print("This will take several minutes (~2-5 GB)...\n")

try:
    import s3fs
    import pickle5

    # Create local cache directory
    cache_dir = "/tmp/scirepeval_cache"
    os.makedirs(cache_dir, exist_ok=True)

    files_to_download = {
        "train_mined_threshold.pkl": "s3://ai2-s2-aps/scirepeval_v2/citation_triplets/checkpoints/train_mined_threshold.pkl",
        "paper_texts.pkl": "s3://ai2-s2-aps/scirepeval_v2/citation_triplets/checkpoints/paper_texts.pkl",
    }

    fs = s3fs.S3FileSystem(anon=False)

    for filename, s3_path in files_to_download.items():
        local_path = os.path.join(cache_dir, filename)

        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024**2)
            print(f"✓ {filename} already cached ({size_mb:.0f} MB)")
            continue

        print(f"Downloading {filename}...")
        print(f"  From: {s3_path}")
        print(f"  To: {local_path}")

        with fs.open(s3_path, "rb") as src:
            with open(local_path, "wb") as dst:
                chunk_size = 10 * 1024 * 1024  # 10 MB chunks
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    downloaded_mb = os.path.getsize(local_path) / (1024**2)
                    print(f"    Downloaded {downloaded_mb:.0f} MB...", end='\r')

        size_mb = os.path.getsize(local_path) / (1024**2)
        print(f"✓ {filename} downloaded ({size_mb:.0f} MB)")

    print(f"\n✓ All files cached in {cache_dir}")
    print(f"Update presample_datasets.py citation pkl paths to use local files:")
    for filename in files_to_download:
        print(f"  {os.path.join(cache_dir, filename)}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
