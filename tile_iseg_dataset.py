"""
tile_iseg_dataset.py
====================
Convert iSeg (.h5 + _resampled.jpg) files into 256x256 nnUNet tiles
and append them to the existing dataset and tile_index.json.

Source folder: SF_iseg_resampled
  Per case:
    <stem>_resampled.jpg   → image (already matches label size)
    <stem>_SF1.h5          → iSeg project (Tissue labels, Blosc-compressed)

iSeg → nnUNet class mapping (read per-file from h5 Tissues metadata):
  Background (iSeg 0)    →  0  Background
  Perineurium            →  1  ConnectivePerineurium
  Epineurium             →  1  ConnectivePerineurium
  Fat                    →  2  Adipose
  Fascicle               →  3  NerveFascicle
  Blood_vessel           →  4  Blood_vessel
  temp / unknown         →  0  (treated as background)

Output:
  imagesTs/<CASE>_0000.tif   (grayscale)
  labelsTs/<CASE>.tif        (integer class labels 0-4)
  tile_index.json            (updated with test entries, split="test")

Requires:
  pip install hdf5plugin     (for Blosc decompression of iSeg h5 files)

Usage:
  python tile_iseg_dataset.py
  python tile_iseg_dataset.py --src /other/folder --split test
  python tile_iseg_dataset.py --split train   # add to training set instead
"""

import argparse
import json
import os

import hdf5plugin   # must be imported before h5py to register Blosc
import h5py
import numpy as np
import tifffile
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SRC = (
    "/home/moyasans/Documents/Data/NerveSegmentation/SF_iseg_resampled"
)
DEFAULT_DATASET = os.path.join(
    _WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI"
)

# ── Settings ──────────────────────────────────────────────────────────────────
TILE_SIZE   = 256
CASE_PREFIX = "NerveMAVI"

# Canonical iSeg tissue-name → nnUNet class index
# Names are matched case-insensitively.
TISSUE_NAME_TO_CLASS = {
    "fascicle"     : 3,   # NerveFascicle
    "perineurium"  : 1,   # ConnectivePerineurium
    "epineurium"   : 1,   # ConnectivePerineurium
    "fat"          : 2,   # Adipose
    "blood_vessel" : 4,   # Blood_vessel
    "temp"         : 0,   # unlabelled / scratch — treat as background
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_remap(h5_file: h5py.File) -> dict:
    """
    Return {iseg_index: nnunet_class} for every tissue in this h5 file.
    iSeg index 0 (background) always maps to 0.
    """
    remap = {0: 0}
    tissues_grp = h5_file["Tissues"]
    for tname in tissues_grp.keys():
        grp = tissues_grp[tname]
        if not hasattr(grp, "keys") or "index" not in grp:
            continue
        iseg_idx   = int(grp["index"][0])
        nnunet_cls = TISSUE_NAME_TO_CLASS.get(tname.lower(), 0)
        remap[iseg_idx] = nnunet_cls
    return remap


def remap_tissue(tissue_flat: np.ndarray, remap: dict) -> np.ndarray:
    """Apply remap dict to a flat or 2-D tissue array."""
    out = np.zeros_like(tissue_flat, dtype=np.uint8)
    for src, dst in remap.items():
        out[tissue_flat == src] = dst
    return out


def count_tiles(H: int, W: int, tile: int) -> int:
    import math
    return math.ceil(H / tile) * math.ceil(W / tile)


def pad_tile(arr: np.ndarray, tile: int) -> np.ndarray:
    """Zero-pad a 2-D tile to (tile, tile) if it is smaller on any edge."""
    h, w = arr.shape
    if h == tile and w == tile:
        return arr
    out = np.zeros((tile, tile), dtype=arr.dtype)
    out[:h, :w] = arr
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src",       default=DEFAULT_SRC,
                        help="Folder containing _resampled.jpg and _SF1.h5 pairs")
    parser.add_argument("--dataset",   default=DEFAULT_DATASET,
                        help="nnUNet dataset root")
    parser.add_argument("--split",     default="test", choices=["train", "test"],
                        help="'test' (default) → imagesTs/labelsTs  |  'train' → imagesTr/labelsTr")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Starting case index (default: auto from tile_index.json)")
    args = parser.parse_args()

    is_train = (args.split == "train")

    # Directories
    images_dir = os.path.join(args.dataset, "imagesTr" if is_train else "imagesTs")
    labels_dir = os.path.join(args.dataset, "labelsTr" if is_train else "labelsTs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Load / init tile index
    index_path = os.path.join(args.dataset, "tile_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            tile_index = json.load(f)
    else:
        tile_index = {}

    # Starting case index
    if args.start_idx is not None:
        case_idx = args.start_idx
    elif tile_index:
        existing = [
            int(k.split("_")[-1])
            for k in tile_index
            if k.startswith(CASE_PREFIX + "_")
        ]
        case_idx = max(existing) + 1 if existing else 0
    else:
        case_idx = 0

    # Collect cases: find all _SF1.h5 files
    h5_files = sorted(
        f for f in os.listdir(args.src) if f.endswith("_SF1.h5")
    )

    print(f"Source folder  : {args.src}")
    print(f"Images output  : {images_dir}")
    print(f"Labels output  : {labels_dir}")
    print(f"Tile index     : {index_path}")
    print(f"Tile size      : {TILE_SIZE}x{TILE_SIZE}")
    print(f"Split          : {args.split}")
    print(f"Start case idx : {case_idx}")
    print(f"Cases found    : {len(h5_files)}")
    print()

    skipped     = 0
    total_tiles = 0

    for h5fname in h5_files:
        # stem: everything before "_SF1.h5"
        stem     = h5fname[: -len("_SF1.h5")]
        h5_path  = os.path.join(args.src, h5fname)
        img_path = os.path.join(args.src, stem + "_resampled.jpg")

        if not os.path.exists(img_path):
            # Fallback: try without _resampled suffix
            img_path = os.path.join(args.src, stem + ".jpg")
        if not os.path.exists(img_path):
            print(f"  SKIP (no image): {h5fname}")
            skipped += 1
            continue

        # ── Load image ───────────────────────────────────────────────────────
        orig_arr = np.array(Image.open(img_path).convert("L"))  # grayscale
        H_img, W_img = orig_arr.shape

        # ── Load label from h5 ───────────────────────────────────────────────
        with h5py.File(h5_path, "r") as f:
            dims    = f["dimensions"][()]     # [X, Y, Z]
            W_lbl   = int(dims[0])
            H_lbl   = int(dims[1])
            tissue  = f["Tissue"][()]         # flat uint16, length H*W
            remap   = build_remap(f)

        if (H_img, W_img) != (H_lbl, W_lbl):
            print(f"  SKIP (size mismatch img={W_img}x{H_img} lbl={W_lbl}x{H_lbl}): {h5fname}")
            skipped += 1
            continue

        label_arr = remap_tissue(tissue, remap).reshape(H_lbl, W_lbl)

        n = count_tiles(H_img, W_img, TILE_SIZE)
        print(
            f"  {h5fname}  [{H_img}x{W_img}]  remap={remap}  → {n} tiles"
            f"  (cases {case_idx}–{case_idx + n - 1})"
        )

        # ── Tile ─────────────────────────────────────────────────────────────
        tile_row = 0
        for y in range(0, H_img, TILE_SIZE):
            tile_col = 0
            for x in range(0, W_img, TILE_SIZE):
                case_name = f"{CASE_PREFIX}_{case_idx:04d}"
                ah = min(TILE_SIZE, H_img - y)  # actual pixels in this tile
                aw = min(TILE_SIZE, W_img - x)

                tifffile.imwrite(
                    os.path.join(images_dir, f"{case_name}_0000.tif"),
                    pad_tile(orig_arr[y : y + ah, x : x + aw], TILE_SIZE),
                )
                tifffile.imwrite(
                    os.path.join(labels_dir, f"{case_name}.tif"),
                    pad_tile(label_arr[y : y + ah, x : x + aw], TILE_SIZE),
                )

                tile_index[case_name] = {
                    "source_file"   : stem + "_resampled.jpg",
                    "source_height" : H_img,
                    "source_width"  : W_img,
                    "tile_row"      : tile_row,
                    "tile_col"      : tile_col,
                    "y_offset"      : y,
                    "x_offset"      : x,
                    "tile_size"     : TILE_SIZE,
                    "actual_h"      : ah,
                    "actual_w"      : aw,
                    "split"         : args.split,
                }

                case_idx   += 1
                total_tiles += 1
                tile_col   += 1
            tile_row += 1

    # Save updated tile index
    with open(index_path, "w") as f:
        json.dump(tile_index, f, indent=2)

    print()
    print(f"{'='*50}")
    print(f"Total tiles generated  : {total_tiles}")
    print(f"Cases skipped          : {skipped}")
    print(f"Tile index saved to    : {index_path}")
    print(f"{'='*50}")
    print()
    if not is_train:
        print("Next steps:")
        print("  1. Run inference on imagesTs/ after training")
        print("  2. Use tile_index.json to reconstruct full predictions")
        print("  3. Compare labelsTs/ against reconstructed predictions for evaluation")


if __name__ == "__main__":
    main()
