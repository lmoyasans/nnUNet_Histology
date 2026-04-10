"""
reconstruct_predictions.py
==========================
Reconstruct full-size prediction images from 256×256 tile predictions
using the spatial metadata in tile_index.json.

For each source image in the test split the script:
  1. Finds all tiles that came from that image
  2. Creates a blank canvas of the original image size
  3. Pastes each predicted tile at its recorded (y_offset, x_offset)
  4. Saves the reconstructed label map as a .tif alongside the original image

Output structure:
  <output_dir>/
    <source_stem>_pred.tif     ← reconstructed label map  (uint8)
    <source_stem>_orig.tif     ← copy of the original image (uint8, grayscale)

Usage:
  python reconstruct_predictions.py \\
      --predictions nnUNet_results/.../fold_0/predictions \\
      --output      nnUNet_results/.../fold_0/reconstructed \\
      [--index      nnUNet_raw/Dataset001_NerveMAVI/tile_index.json] \\
      [--raw        nnUNet_raw/Dataset001_NerveMAVI] \\
      [--label      CE_Histo]
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import tifffile
from PIL import Image

# ── Defaults ──────────────────────────────────────────────────────────────────
_WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(_WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI")
INDEX_PATH   = os.path.join(_WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI", "tile_index.json")


def find_source_image(source_file: str, raw_dir: str) -> str | None:
    """Locate the original image file (jpg or tif) for a given source_file name."""
    stem      = os.path.splitext(source_file)[0]
    # For GTEx tiles the source_file IS the image file name (a .tif)
    # For iSeg tiles the source_file is <stem>_resampled.jpg
    # Try raw dataset folder and known data folders
    search_roots = [
        raw_dir,
        os.path.join(_WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI", "imagesTs"),
        "/home/moyasans/Documents/Data/NerveSegmentation/Tibial segmentations/Goldstandards/Smoothed_GoldStandard_GTExTibial",
        "/home/moyasans/Documents/Data/NerveSegmentation/SF_iseg_resampled",
    ]
    for root in search_roots:
        candidate = os.path.join(root, source_file)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="Folder containing predicted .tif tiles")
    parser.add_argument("--output",      required=True,
                        help="Folder to write reconstructed images")
    parser.add_argument("--index",       default=INDEX_PATH,
                        help="Path to tile_index.json")
    parser.add_argument("--raw",         default=DATASET_DIR,
                        help="nnUNet raw dataset directory")
    parser.add_argument("--label",       default="",
                        help="Short experiment label for display (e.g. CE_Histo)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load tile index
    with open(args.index) as f:
        tile_index = json.load(f)

    # Build: source_file → {case_name: tile_meta} for test-split tiles
    source_to_tiles = defaultdict(dict)
    for case_name, meta in tile_index.items():
        if meta["split"] == "test":
            source_to_tiles[meta["source_file"]][case_name] = meta

    print(f"{'='*55}")
    print(f"Experiment     : {args.label or '(no label)'}")
    print(f"Predictions    : {args.predictions}")
    print(f"Output         : {args.output}")
    print(f"Source images  : {len(source_to_tiles)}")
    print(f"{'='*55}")

    for source_file, tiles in sorted(source_to_tiles.items()):
        # All tiles share the same source dimensions
        first_meta = next(iter(tiles.values()))
        H          = first_meta["source_height"]
        W          = first_meta["source_width"]
        tile_size  = first_meta["tile_size"]

        # Blank canvas
        canvas = np.zeros((H, W), dtype=np.uint8)
        placed = 0
        missing = 0

        for case_name, meta in tiles.items():
            pred_path = os.path.join(args.predictions, f"{case_name}.tif")
            if not os.path.exists(pred_path):
                missing += 1
                continue
            tile_pred = np.array(Image.open(pred_path))
            y  = meta["y_offset"]
            x  = meta["x_offset"]
            ah = meta.get("actual_h", tile_size)  # actual content rows
            aw = meta.get("actual_w", tile_size)  # actual content cols
            canvas[y : y + ah, x : x + aw] = tile_pred[:ah, :aw]
            placed += 1

        # Save reconstructed prediction
        stem      = os.path.splitext(source_file)[0]
        pred_out  = os.path.join(args.output, f"{stem}_pred.tif")
        tifffile.imwrite(pred_out, canvas)

        # Save copy of original image (grayscale) alongside prediction
        orig_path = find_source_image(source_file, args.raw)
        orig_out  = os.path.join(args.output, f"{stem}_orig.tif")
        if orig_path and not os.path.exists(orig_out):
            orig_arr = np.array(Image.open(orig_path).convert("L"))
            tifffile.imwrite(orig_out, orig_arr)
            orig_info = f"original saved"
        elif os.path.exists(orig_out):
            orig_info = f"original already exists"
        else:
            orig_info = f"WARNING: original not found ({source_file})"

        print(f"  {stem}")
        print(f"    canvas {H}×{W}  placed={placed}  missing={missing}")
        print(f"    pred  → {os.path.basename(pred_out)}")
        print(f"    {orig_info}")

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
