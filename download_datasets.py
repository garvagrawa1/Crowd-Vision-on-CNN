"""Download and unpack the two public datasets used to train CrowdVisionNet.

  datasets/lfw/lfw-deepfunneled/...           ~13,000 face photos (Labeled Faces in the Wild)
  datasets/caltech101/101_ObjectCategories/... ~9,000 object/scene photos (non-face examples)

Both are fetched from public GitHub mirrors (~260 MB in total). If the .zip
files are already in datasets/, they are just unpacked.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "lfw": ("https://codeload.github.com/Arun-no-code/lfww/zip/refs/heads/main", "lfw.zip", "lfw-deepfunneled"),
    "caltech101": ("https://codeload.github.com/hrishavkmr/Caltech101_Dataset/zip/refs/heads/master",
                   "caltech101.zip", "101_ObjectCategories"),
}
ROOT = Path("datasets")


def progress(block, block_size, total):
    done = block * block_size / 1e6
    sys.stdout.write(f"\r   downloaded {done:6.1f} MB"); sys.stdout.flush()


def main():
    ROOT.mkdir(exist_ok=True)
    for name, (url, zname, inner) in SOURCES.items():
        target = ROOT / name / inner
        if target.exists() and any(target.iterdir()):
            print(f"[ok] {target} already present")
            continue
        zpath = ROOT / zname
        if not zpath.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, zpath, progress); print()
        print(f"Unpacking {zpath} ...")
        tmp = ROOT / f"_tmp_{name}"
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        found = next(tmp.rglob(inner))
        (ROOT / name).mkdir(exist_ok=True)
        shutil.move(str(found), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[ok] {target}")
    print("\nDatasets ready. Next:  python prepare_data.py")


if __name__ == "__main__":
    main()
