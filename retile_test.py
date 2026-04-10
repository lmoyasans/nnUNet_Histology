"""
retile_test.py
==============
Remove old (edge-missing) test tiles and re-tile all 4 test images with
proper edge tile support (padded partial tiles at image borders).

Test images:
  GTEx:      GTEX-1212Z-2725_2.tif  +  GTEX-SSA3-0525_2.tif
             (source: Smoothed_GoldStandard_GTExTibial/)
  Bio-Aegis: O21574 + O22114  (source: SF_iseg_resampled/)

Strategy:
  1. Remove all test entries from tile_index.json (+ their image/label files)
  2. Re-tile GTEx test images directly (using tile_dataset logic)
  3. Re-tile Bio-Aegis test images (using tile_iseg_dataset logic)
  New tiles start at max_train_idx + 1 to avoid any index collision.

Usage:
  python retile_test.py [--dry-run]
"""

import argparse
import json
import math
import os

import hdf5plugin  # must be before h5py
import h5py
import numpy as np
import tifffile
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE, 'nnUNet_raw', 'Dataset001_NerveMAVI')

GTEX_SRC = os.path.join(
    '/home/moyasans/Documents/Data/NerveSegmentation',
    'Tibial segmentations/Goldstandards/Smoothed_GoldStandard_GTExTibial',
)
ISEG_SRC = os.path.join(
    '/home/moyasans/Documents/Data/NerveSegmentation',
    'SF_iseg_resampled',
)

# Specific test files
GTEX_TEST_STEMS = ['GTEX-1212Z-2725_2', 'GTEX-SSA3-0525_2']
ISEG_TEST_STEMS = [
    'Bio-Aegis-H-25-01_O21574_Bl.13_1um_HE_overview_40x_resampled',
    'Bio-Aegis-H-25-01_O22114_Bl.12_1um_HE_overview_10x_resampled',
]

TILE_SIZE   = 256
CASE_PREFIX = 'NerveMAVI'

# GTEx label mapping
GTEX_LABEL_MAP = {0: 0, 63: 1, 127: 2, 255: 3}

# iSeg tissue-name → nnUNet class
TISSUE_NAME_TO_CLASS = {
    'fascicle':    3,
    'perineurium': 1,
    'epineurium':  1,
    'fat':         2,
    'blood_vessel':4,
    'temp':        0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_tile(arr: np.ndarray, tile: int) -> np.ndarray:
    h, w = arr.shape
    if h == tile and w == tile:
        return arr
    out = np.zeros((tile, tile), dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def remap_gtex(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for src, dst in GTEX_LABEL_MAP.items():
        out[arr == src] = dst
    return out


def build_iseg_remap(h5_file: h5py.File) -> dict:
    remap = {0: 0}
    tissues_grp = h5_file['Tissues']
    for tname in tissues_grp.keys():
        grp = tissues_grp[tname]
        if not hasattr(grp, 'keys') or 'index' not in grp:
            continue
        iseg_idx   = int(grp['index'][0])
        nnunet_cls = TISSUE_NAME_TO_CLASS.get(tname.lower(), 0)
        remap[iseg_idx] = nnunet_cls
    return remap


def remap_tissue(tissue_flat: np.ndarray, remap: dict) -> np.ndarray:
    out = np.zeros_like(tissue_flat, dtype=np.uint8)
    for src, dst in remap.items():
        out[tissue_flat == src] = dst
    return out


def tile_image(orig_arr: np.ndarray, label_arr: np.ndarray,
               source_name: str, case_idx: int,
               images_dir: str, labels_dir: str,
               tile_index: dict, dry_run: bool) -> int:
    """Tile one image+label pair with edge support. Returns new case_idx."""
    H, W = orig_arr.shape[:2]
    n = math.ceil(H / TILE_SIZE) * math.ceil(W / TILE_SIZE)
    print(f'  [{H}x{W}] → {n} tiles  (cases {case_idx}–{case_idx+n-1})')

    tile_row = 0
    for y in range(0, H, TILE_SIZE):
        tile_col = 0
        for x in range(0, W, TILE_SIZE):
            case_name = f'{CASE_PREFIX}_{case_idx:04d}'
            ah = min(TILE_SIZE, H - y)
            aw = min(TILE_SIZE, W - x)

            if not dry_run:
                tifffile.imwrite(
                    os.path.join(images_dir, f'{case_name}_0000.tif'),
                    pad_tile(orig_arr[y:y+ah, x:x+aw], TILE_SIZE),
                )
                tifffile.imwrite(
                    os.path.join(labels_dir, f'{case_name}.tif'),
                    pad_tile(label_arr[y:y+ah, x:x+aw], TILE_SIZE),
                )

            tile_index[case_name] = {
                'source_file':    source_name,
                'source_height':  H,
                'source_width':   W,
                'tile_row':       tile_row,
                'tile_col':       tile_col,
                'y_offset':       y,
                'x_offset':       x,
                'tile_size':      TILE_SIZE,
                'actual_h':       ah,
                'actual_w':       aw,
                'split':          'test',
            }
            case_idx += 1
            tile_col += 1
        tile_row += 1
    return case_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without writing files')
    args = parser.parse_args()

    index_path = os.path.join(DATASET_DIR, 'tile_index.json')
    images_ts  = os.path.join(DATASET_DIR, 'imagesTs')
    labels_ts  = os.path.join(DATASET_DIR, 'labelsTs')

    # ── Load tile index ───────────────────────────────────────────────────────
    tile_index = json.load(open(index_path))

    # ── Step 1: remove old test tiles ─────────────────────────────────────────
    old_test = [k for k, v in tile_index.items() if v.get('split') == 'test']
    print(f'Removing {len(old_test)} old test tile entries...')
    for case_name in old_test:
        img_path = os.path.join(images_ts, f'{case_name}_0000.tif')
        lbl_path = os.path.join(labels_ts, f'{case_name}.tif')
        if not args.dry_run:
            for p in (img_path, lbl_path):
                if os.path.exists(p):
                    os.remove(p)
        del tile_index[case_name]

    # ── Determine start index (one past max training idx) ────────────────────
    train_indices = [int(k.split('_')[-1]) for k in tile_index
                     if k.startswith(CASE_PREFIX + '_')]
    case_idx = max(train_indices) + 1 if train_indices else 0
    print(f'New test tiles start at idx: {case_idx}\n')

    os.makedirs(images_ts, exist_ok=True)
    os.makedirs(labels_ts, exist_ok=True)

    # ── Step 2: Re-tile GTEx test images ──────────────────────────────────────
    print('=== GTEx test images ===')
    for stem in GTEX_TEST_STEMS:
        img_path = os.path.join(GTEX_SRC, f'{stem}.tif')
        lbl_path = os.path.join(GTEX_SRC, f'_{stem}_smooth.tif')
        if not os.path.exists(img_path):
            print(f'  SKIP (no image): {stem}')
            continue
        if not os.path.exists(lbl_path):
            print(f'  SKIP (no label): {stem}')
            continue

        orig_arr  = np.array(Image.open(img_path).convert('L'))
        label_arr = remap_gtex(np.array(Image.open(lbl_path)))
        print(f'  {stem}')
        case_idx = tile_image(orig_arr, label_arr,
                              f'{stem}.tif', case_idx,
                              images_ts, labels_ts,
                              tile_index, args.dry_run)

    # ── Step 3: Re-tile Bio-Aegis test images ──────────────────────────────────
    print('\n=== Bio-Aegis test images ===')
    h5_files = sorted(f for f in os.listdir(ISEG_SRC) if f.endswith('_SF1.h5'))
    for h5fname in h5_files:
        stem     = h5fname[:-len('_SF1.h5')]
        src_name = stem + '_resampled.jpg'
        if not any(s in stem for s in ['O21574', 'O22114']):
            continue  # only tile the two test cases

        img_path = os.path.join(ISEG_SRC, stem + '_resampled.jpg')
        if not os.path.exists(img_path):
            print(f'  SKIP (no image): {h5fname}')
            continue

        orig_arr = np.array(Image.open(img_path).convert('L'))
        H_img, W_img = orig_arr.shape

        with h5py.File(os.path.join(ISEG_SRC, h5fname), 'r') as f:
            dims   = f['dimensions'][()]
            W_lbl, H_lbl = int(dims[0]), int(dims[1])
            tissue = f['Tissue'][()]
            remap  = build_iseg_remap(f)

        if (H_img, W_img) != (H_lbl, W_lbl):
            print(f'  SKIP (size mismatch): {h5fname}')
            continue

        label_arr = remap_tissue(tissue, remap).reshape(H_lbl, W_lbl)
        print(f'  {stem}')
        case_idx = tile_image(orig_arr, label_arr,
                              src_name, case_idx,
                              images_ts, labels_ts,
                              tile_index, args.dry_run)

    # ── Save updated tile index ───────────────────────────────────────────────
    if not args.dry_run:
        with open(index_path, 'w') as f:
            json.dump(tile_index, f, indent=2)
        print(f'\ntile_index.json updated → {index_path}')
    else:
        print('\n[DRY RUN] No files written.')

    new_test = [k for k, v in tile_index.items() if v.get('split') == 'test']
    print(f'Total test tiles now: {len(new_test)}')


if __name__ == '__main__':
    main()
