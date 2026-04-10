"""
tile_dataset.py
===============
Generate 256x256 non-overlapping tiles from the Tibial nerve dataset
and save them in nnUNet format.

Source folder: Smoothed_GoldStandard_GTExTibial
  - Original images : GTEX-XXXXX.tif          (RGB)
  - Smoothed labels : _GTEX-XXXXX_smooth.tif  (grayscale, values {0,63,127,255})

Output:
  - imagesTr/NerveMAVI_XXXX_0000.tif  (grayscale uint8)
  - labelsTr/NerveMAVI_XXXX.tif       (integer labels {0,1,2,3})
  - tile_index.json                   (mapping for reconstruction)

Label mapping (pixel value → class index):
  0   → 0  Background
  63  → 1  ConnectivePerineurium
  127 → 2  Adipose
  255 → 3  NerveFascicle

tile_index.json structure:
  A dict keyed by case name (e.g. "NerveMAVI_0042") with entries:
    {
      "source_file"   : "GTEX-1212Z-2725_2.tif",  # original image filename
      "source_height" : 6428,                       # full image height (pixels)
      "source_width"  : 5980,                       # full image width  (pixels)
      "tile_row"      : 2,                          # 0-based row index of tile
      "tile_col"      : 3,                          # 0-based col index of tile
      "y_offset"      : 512,                        # top-left pixel y in original
      "x_offset"      : 768,                        # top-left pixel x in original
      "tile_size"     : 256,                        # tile side length
      "split"         : "train"                     # "train" or "test"
    }

  Use x_offset / y_offset to place a predicted tile back into a blank canvas
  of size (source_height, source_width).

NOTE on RGB vs grayscale:
  The source images are RGB but dataset.json is configured for a single
  'Histology' channel.  Images are therefore converted to grayscale.
  If you want to use RGB, update dataset.json with 3 channel entries
  ("0":"R","1":"G","2":"B") and change KEEP_RGB = True — the script will
  then save three _0000/_0001/_0002 channel files per tile.

Usage:
  # Training tiles (default)
  python tile_dataset.py

  # Test tiles (append to same index, output to imagesTs)
  python tile_dataset.py --src /path/to/test/images --split test \\
      --start-idx 7011

  Both calls share the same tile_index.json so reconstruction always
  works regardless of split.
"""

import argparse
import json
import os

import numpy as np
import tifffile
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SRC = os.path.join(
    "/home/moyasans/Documents/Data/NerveSegmentation",
    "Tibial segmentations/Goldstandards/Smoothed_GoldStandard_GTExTibial",
)
DEFAULT_DATASET = os.path.join(
    _WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI"
)

# ── Settings ──────────────────────────────────────────────────────────────────
TILE_SIZE   = 256
CASE_PREFIX = "NerveMAVI"
KEEP_RGB    = False   # False → grayscale (matches single-channel dataset.json)

LABEL_MAP = {0: 0, 63: 1, 127: 2, 255: 3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_image(arr: np.ndarray, method: str, scale_file: str = None) -> np.ndarray:
    """
    Normalize a full-resolution grayscale image *before* tiling.

    Normalizing on the whole image (rather than per-tile) ensures that
    intensity statistics are computed from a complete tissue context and
    are consistent across all tiles that originate from the same slide.

    method:
      'none'       – return unchanged (uint8 preserved)
      'histology'  – clip [1 %, 99 %] percentile, rescale to [0, 1] (float32)
      'nyul'       – Nyul & Udupa landmark standardisation to [0, 1] (float32)
                     Requires scale_file pointing to nyul_standard_scale.json.
                     Run compute_nyul_scale.py once to generate it.
    """
    if method == 'none':
        return arr

    img = arr.astype(np.float32)

    if method == 'histology':
        p_low, p_high = np.percentile(img, [1.0, 99.0])
        if p_high > p_low:
            np.clip(img, p_low, p_high, out=img)
            img -= p_low
            img /= (p_high - p_low)
        return img

    if method == 'nyul':
        if scale_file is None or not os.path.exists(scale_file):
            raise FileNotFoundError(
                f"Nyul standard scale file not found: {scale_file}\n"
                "Run: python compute_nyul_scale.py"
            )
        with open(scale_file) as _f:
            _data = json.load(_f)
        _percentiles    = np.array(_data['percentiles'],    dtype=np.float64)
        _standard_scale = np.array(_data['standard_scale'], dtype=np.float64)
        _img_landmarks  = np.percentile(img, _percentiles)
        img_out = np.interp(img.ravel(), _img_landmarks, _standard_scale)
        img_out = img_out.reshape(img.shape).astype(np.float32)
        np.clip(img_out, 0.0, 1.0, out=img_out)
        return img_out

    raise ValueError(f"Unknown normalization method: {method!r}")


def remap_labels(arr: np.ndarray) -> np.ndarray:
    """Map raw pixel values {0,63,127,255} → class indices {0,1,2,3}."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    for src_val, dst_val in LABEL_MAP.items():
        out[arr == src_val] = dst_val
    return out


def count_tiles(H: int, W: int, tile: int) -> int:
    import math
    return math.ceil(H / tile) * math.ceil(W / tile)


def pad_tile(arr: np.ndarray, tile: int) -> np.ndarray:
    """Zero-pad a 2-D tile to (tile, tile) if smaller on any edge."""
    h, w = arr.shape
    if h == tile and w == tile:
        return arr
    out = np.zeros((tile, tile), dtype=arr.dtype)
    out[:h, :w] = arr
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",       default=DEFAULT_SRC,
                        help="Folder containing original (+ smoothed) .tif files")
    parser.add_argument("--dataset",   default=DEFAULT_DATASET,
                        help="nnUNet dataset root (imagesTr/, labelsTr/, imagesTs/)")
    parser.add_argument("--split",     default="train", choices=["train", "test"],
                        help="Which split to generate: 'train' (default) or 'test'")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Case index to start from (default: auto-detect from tile_index.json)")
    parser.add_argument(
        "--norm", default="none", choices=["none", "histology", "nyul"],
        help="Normalization applied to the FULL image before tiling "
             "(default: none).  Pair with NoNormalization in the nnUNet plan "
             "so tiles are not re-normalized per-patch during preprocessing.",
    )
    parser.add_argument(
        "--scale-file",
        default=os.path.join(
            _WORKSPACE, "nnUNet_preprocessed", "Dataset001_NerveMAVI",
            "nyul_standard_scale.json",
        ),
        help="Path to nyul_standard_scale.json (only used with --norm nyul)",
    )
    args = parser.parse_args()

    is_train = (args.split == "train")

    # Output directories
    if is_train:
        images_dir = os.path.join(args.dataset, "imagesTr")
        labels_dir = os.path.join(args.dataset, "labelsTr")
        os.makedirs(labels_dir, exist_ok=True)
    else:
        images_dir = os.path.join(args.dataset, "imagesTs")
    os.makedirs(images_dir, exist_ok=True)

    # Load (or initialise) tile index
    index_path = os.path.join(args.dataset, "tile_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            tile_index = json.load(f)
    else:
        tile_index = {}

    # Determine starting case index
    if args.start_idx is not None:
        case_idx = args.start_idx
    elif tile_index:
        # Auto-detect: max existing index + 1
        existing_indices = [
            int(k.split("_")[-1]) for k in tile_index if k.startswith(CASE_PREFIX + "_")
        ]
        case_idx = max(existing_indices) + 1 if existing_indices else 0
    else:
        case_idx = 0

    # Collect source files
    # For training: expect paired original + _smooth files
    # For test:     expect only original files (no label required)
    all_files  = os.listdir(args.src)
    orig_files = sorted(
        f for f in all_files
        if not f.startswith("_") and f.endswith(".tif")
    )

    print(f"Source folder  : {args.src}")
    print(f"Output images  : {images_dir}")
    if is_train:
        print(f"Output labels  : {labels_dir}")
    print(f"Tile index     : {index_path}")
    print(f"Tile size      : {TILE_SIZE}x{TILE_SIZE}")
    print(f"Split          : {args.split}")
    print(f"Start case idx : {case_idx}")
    print(f"Image mode     : {'RGB (3 channels)' if KEEP_RGB else 'Grayscale (1 channel)'}")
    print(f"Normalization  : {args.norm}")
    print(f"Originals found: {len(orig_files)}")
    print()

    skipped     = 0
    total_tiles = 0

    for orig_name in orig_files:
        stem      = os.path.splitext(orig_name)[0]
        orig_path = os.path.join(args.src, orig_name)

        # ── Load image ───────────────────────────────────────────────────────
        pil_orig = Image.open(orig_path)
        if KEEP_RGB:
            orig_arr = np.array(pil_orig.convert("RGB"))
            if args.norm != 'none':
                print(f"  WARNING: --norm '{args.norm}' is ignored when KEEP_RGB=True")
        else:
            # Normalize the FULL image before tiling so that intensity
            # statistics are derived from the whole slide, not per-patch.
            orig_arr = normalize_image(
                np.array(pil_orig.convert("L")), args.norm, args.scale_file
            )

        H, W = orig_arr.shape[:2]

        # ── Load label (train only) ──────────────────────────────────────────
        if is_train:
            label_name = f"_{stem}_smooth.tif"
            label_path = os.path.join(args.src, label_name)
            if not os.path.exists(label_path):
                print(f"  SKIP (no label): {orig_name}")
                skipped += 1
                continue
            label_arr = remap_labels(np.array(Image.open(label_path)))
            if label_arr.shape != (H, W):
                print(f"  SKIP (shape mismatch): {orig_name}")
                skipped += 1
                continue

        n = count_tiles(H, W, TILE_SIZE)
        print(f"  {orig_name}  [{H}x{W}]  → {n} tiles  (cases {case_idx}–{case_idx + n - 1})")

        # ── Tile ─────────────────────────────────────────────────────────────
        tile_row = 0
        for y in range(0, H, TILE_SIZE):
            tile_col = 0
            for x in range(0, W, TILE_SIZE):
                case_name = f"{CASE_PREFIX}_{case_idx:04d}"
                ah = min(TILE_SIZE, H - y)  # actual pixels in this tile
                aw = min(TILE_SIZE, W - x)

                # Save image tile
                img_tile = orig_arr[y : y + ah, x : x + aw]
                if KEEP_RGB:
                    img_tile_pad = np.zeros((TILE_SIZE, TILE_SIZE, img_tile.shape[2]), dtype=img_tile.dtype)
                    img_tile_pad[:ah, :aw] = img_tile
                    for ch_idx in range(3):
                        tifffile.imwrite(
                            os.path.join(images_dir, f"{case_name}_{ch_idx:04d}.tif"),
                            img_tile_pad[:, :, ch_idx],
                        )
                else:
                    tifffile.imwrite(
                        os.path.join(images_dir, f"{case_name}_0000.tif"),
                        pad_tile(img_tile, TILE_SIZE),
                    )

                # Save label tile (train only)
                if is_train:
                    tifffile.imwrite(
                        os.path.join(labels_dir, f"{case_name}.tif"),
                        pad_tile(label_arr[y : y + ah, x : x + aw], TILE_SIZE),
                    )

                # Record metadata in tile index
                tile_index[case_name] = {
                    "source_file"   : orig_name,
                    "source_height" : H,
                    "source_width"  : W,
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
    print(f"Total tiles generated : {total_tiles}")
    print(f"Source images skipped : {skipped}")
    print(f"Tile index saved to   : {index_path}")
    print(f"{'='*50}")
    print()
    if is_train:
        print("Next steps:")
        print("  1. Update numTraining in dataset.json to match total tiles")
        print("  2. Run: nnUNetv2_plan_and_preprocess -d 1 -c 2d -np 4")
        print("  3. Train: nnUNetv2_train 1 2d 0 -tr nnUNetTrainerAdamEarlyStopping_LowLR")
    else:
        print("Next steps:")
        print("  1. Run inference: nnUNetv2_predict -d 1 -i imagesTs -o predictions -c 2d -f 0")
        print("  2. Use tile_index.json to reconstruct full images from predictions")


if __name__ == "__main__":
    main()
