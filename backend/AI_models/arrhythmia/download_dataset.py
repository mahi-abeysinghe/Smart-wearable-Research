# pyrefly: ignore [missing-import]
import wfdb
import os
import urllib.request
import pandas as pd

def download_mit_bih(target_dir="./data"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Downloading MIT-BIH Arrhythmia Database to {target_dir}...")
    
    # Normally we would use wfdb.dl_database('mitdb', target_dir)
    # This downloads the 48 records. It requires roughly ~70MB.
    try:
        wfdb.dl_database('mitdb', target_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_mit_bih()
