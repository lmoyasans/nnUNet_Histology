from pathlib import Path
import tifffile as tiff  # pip install tifffile

def info(path):
    arr = tiff.imread(path)
    print(f"\n{path.name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")

imagesTr = Path("nnUNet_raw/Dataset001_NerveMAVI/imagesTr/")

imagesTs = Path("nnUNet_raw/Dataset001_NerveMAVI/imagesTs/")

# replace with one tif that worked and one that fails
working = imagesTr / "NERVE_1_0000.tif"
failing = imagesTs / "NERVE_5807_0000.tif"

info(working)
info(failing)

from pathlib import Path
import tifffile as tiff
import numpy as np

in_path = imagesTs / "NERVE_5811_0000.tif"
out_path = imagesTs / "NERVE_5811_0000_gray.tif"  # temp name

img = tiff.imread(in_path)              # (H, W, 3)
gray = img.mean(axis=2).astype(img.dtype)  # simple average to (H, W)

tiff.imwrite(out_path, gray)

