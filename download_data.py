"""Download NSL-KDD dataset files into backend/data/."""

import os
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

FILES = {
    "KDDTrain+.txt": f"{BASE_URL}/KDDTrain%2B.txt",
    "KDDTest+.txt": f"{BASE_URL}/KDDTest%2B.txt",
}


def download():
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  [download] {filename} …")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"  [done] {filename} ({len(resp.content) / 1024:.0f} KB)")


if __name__ == "__main__":
    print("Downloading NSL-KDD dataset …")
    download()
    print("All files ready.")
