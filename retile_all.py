"""
retile_all.py
=============
Rebuild the full nnUNet tile dataset from scratch, assigning each source
image to train or test according to split_config.json.

Key guarantee:
  All 256×256 tiles that come from the same source image are placed in the
  same split folder (imagesTr/labelsTr  OR  imagesTs/labelsTs).
  This means images can be fully reconstructed after prediction.

The tile_index.json written alongside records — for every tile:
  source_file, source_height, source_width,
  tile_row, tile_col, y_offset, x_offset, tile_size, split

Split can be reconfigured by editing split_config.json and re-running this
script.  All tile folders and the old tile_index.json are wiped first.

Supports two source types:
  "gtex"  – TIF image  + TIF label (pixel values 0/63/127/255 → classes 0-3)
  "iseg"  – JPG image  + iSeg .h5 label (Blosc-compressed, class map read
            per-file from Tissues/ metadata)

Label class map:
  GTEx pixel → class:   0→0   63→1   127→2   255→3
  iSeg tissue → class:
    Fascicle      → 3  NerveFascicle
    Perineurium   → 1  ConnectivePerineurium
    Epineurium    → 1  ConnectivePerineurium
    Fat           → 2  Adipose
    Blood_vessel  → 4  Blood_vessel
    temp/unknown  → 0  Background

Usage:
  python retile_all.py
  python retile_all.py --config /other/split_config.json
  python retile_all.py --dry-run    # preview without writing any files
"""

import argparse
import json
import os
import shutil

import numpy as np
import tifffile
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
_WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(_WORKSPACE, "nnUNet_raw", "Dataset001_NerveMAVI")
TILE_SIZE    = 256
CASE_PREFIX  = "NerveMAVI"

GTEX_LABEL_MAP = {0: 0, 63: 1, 127: 2, 255: 3}

ISEG_TISSUE_TO_CLASS = {
    "fascicle"     : 3,
    "perineurium"  : 1,
    "epineurium"   : 1,
    "fat"          : 2,
    "blood_vessel" : 4,
    "temp"         : 0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def clear_dir(path: str):
    """Remove all files inside a directory (keep the directory itself)."""
    if os.path.isdir(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                os.remove(fp)
    else:
        os.makedirs(path, exist_ok=True)


def remap_gtex_label(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for src, dst in GTEX_LABEL_MAP.items():
        out[arr == src] = dst
    return out


def build_iseg_remap(h5_file) -> dict:
    """Read per-file iSeg tissue index → nnUNet class mapping."""
    remap = {0: 0}
    for tname in h5_file["Tissues"].keys():
        grp = h5_file["Tissues"][tname]
        if hasattr(grp, "keys") and "index" in grp:
            iseg_idx = int(grp["index"][0])
            remap[iseg_idx] = ISEG_TISSUE_TO_CLASS.get(tname.lower(), 0)
    return remap


def remap_iseg_tissue(tissue: np.ndarray, remap: dict) -> np.ndarray:
    out = np.zeros(tissue.shape, dtype=np.uint8)
    for src, dst in remap.items():
        out[tissue == src] = dst
    return out


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


def tile_image(orig_arr: np.ndarray, label_arr: np.ndarray,
               source_file: str, split: str,
               case_idx: int, images_dir: str, labels_dir: str,
               tile_index: dict, dry_run: bool) -> int:
    """
    Slice orig_arr and label_arr into TILE_SIZE tiles, write them, update
    tile_index, and return the number of tiles written.
    """
    H, W = orig_arr.shape[:2]
    n_written = 0
    tile_row = 0
    for y in range(0, H - TILE_SIZE + 1, TILE_SIZE):
        tile_col = 0
        for x in range(0, W - TILE_SIZE + 1, TILE_SIZE):
            case_name = f"{CASE_PREFIX}_{case_idx:04d}"
            if not dry_run:
                tifffile.imwrite(
                    os.path.join(images_dir, f"{case_name}_0000.tif"),
                    orig_arr[y : y + TILE_SIZE, x : x + TILE_SIZE],
                )
                tifffile.imwrite(
                    os.path.join(labels_dir, f"{case_name}.tif"),
                    label_arr[y : y + TILE_SIZE, x : x + TILE_SIZE],
                )
            tile_index[case_name] = {
                "source_file"   : source_file,
                "source_height" : H,
                "source_width"  : W,
                "tile_row"      : tile_row,
                "tile_col"      : tile_col,
                "y_offset"      : y,
                "x_offset"      : x,
                "tile_size"     : TILE_SIZE,
                "split"         : split,
            }
            case_idx  += 1
            n_written += 1
            tile_col  += 1
        tile_row += 1
    return n_written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=os.path.join(DATASET_DIR, "split_config.json"),
        help="Path to split_config.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without writing any files",
    )
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

    with open(args.config) as f:
        config = json.load(f)

    sources = config["sources"]

    # Directories
    dirs = {
        "imagesTr" : os.path.join(DATASET_DIR, "imagesTr"),
        "labelsTr" : os.path.join(DATASET_DIR, "labelsTr"),
        "imagesTs" : os.path.join(DATASET_DIR, "imagesTs"),
        "labelsTs" : os.path.join(DATASET_DIR, "labelsTs"),
    }
    index_path = os.path.join(DATASET_DIR, "tile_index.json")

    # ── Wipe previous tiles ──────────────────────────────────────────────────
    if not args.dry_run:
        print("Clearing existing tile folders...")
        for d in dirs.values():
            clear_dir(d)
        # Reset tile index
        tile_index = {}
    else:
        tile_index = {}
        print("[DRY RUN — no files will be written]\n")

    print(f"Config         : {args.config}")
    print(f"Dataset dir    : {DATASET_DIR}")
    print(f"Tile size      : {TILE_SIZE}x{TILE_SIZE}")
    print(f"Normalization  : {args.norm}")
    print(f"Total sources  : {len(sources)}")
    print()

    # Print plan header
    train_src = [s for s in sources if s["split"] == "train"]
    test_src  = [s for s in sources if s["split"] == "test"]
    print(f"  Train sources: {len(train_src)}")
    for s in train_src:
        print(f"    [{s['type']:4s}] {s['image_file']}")
    print(f"  Test sources : {len(test_src)}")
    for s in test_src:
        print(f"    [{s['type']:4s}] {s['image_file']}")
    print()

    # Import hdf5plugin lazily (only needed for iseg)
    hdf5plugin = h5py = None
    if any(s["type"] == "iseg" for s in sources):
        import hdf5plugin as _hdf5plugin
        import h5py as _h5py
        hdf5plugin = _hdf5plugin
        h5py = _h5py

    case_idx    = 0
    total_train = 0
    total_test  = 0

    for entry in sources:
        src_type   = entry["type"]
        folder     = entry["folder"]
        image_file = entry["image_file"]
        split      = entry["split"]

        images_dir = dirs["imagesTr"] if split == "train" else dirs["imagesTs"]
        labels_dir = dirs["labelsTr"] if split == "train" else dirs["labelsTs"]

        img_path = os.path.join(folder, image_file)

        # ── Load image ───────────────────────────────────────────────────────
        # Normalize the FULL image before tiling so that intensity
        # statistics are derived from the whole slide, not per-patch.
        orig_arr = normalize_image(
            np.array(Image.open(img_path).convert("L")), args.norm, args.scale_file
        )
        H, W = orig_arr.shape

        # ── Load label ───────────────────────────────────────────────────────
        if src_type == "gtex":
            label_path = os.path.join(folder, entry["label_file"])
            raw_label  = np.array(Image.open(label_path))
            label_arr  = remap_gtex_label(raw_label)
            remap_info = "GTEx {0,63,127,255}→{0,1,2,3}"

        elif src_type == "iseg":
            h5_path = os.path.join(folder, entry["h5_file"])
            with h5py.File(h5_path, "r") as f:
                dims   = f["dimensions"][()]
                tissue = f["Tissue"][()]
                remap  = build_iseg_remap(f)
            label_arr  = remap_iseg_tissue(tissue, remap).reshape(int(dims[1]), int(dims[0]))
            remap_info = str(remap)

        else:
            print(f"  SKIP unknown type '{src_type}': {image_file}")
            continue

        if label_arr.shape != (H, W):
            print(f"  SKIP size mismatch img={W}x{H} lbl={label_arr.shape}: {image_file}")
            continue

        n_tiles = (H // TILE_SIZE) * (W // TILE_SIZE)
        print(
            f"  [{split:5s}] [{src_type:4s}]  {image_file}\n"
            f"           {H}×{W}  →  {n_tiles} tiles"
            f"  (cases {case_idx}–{case_idx + n_tiles - 1})"
        )

        n_written = tile_image(
            orig_arr, label_arr, image_file, split,
            case_idx, images_dir, labels_dir,
            tile_index, dry_run=args.dry_run,
        )
        case_idx += n_written
        if split == "train":
            total_train += n_written
        else:
            total_test  += n_written

    # ── Save tile index ──────────────────────────────────────────────────────
    if not args.dry_run:
        with open(index_path, "w") as f:
            json.dump(tile_index, f, indent=2)

    print()
    print("=" * 55)
    print(f"Train tiles  : {total_train}")
    print(f"Test tiles   : {total_test}")
    print(f"Total tiles  : {case_idx}")
    print(f"Tile index   : {index_path}")
    print("=" * 55)
    print()
    if not args.dry_run:
        print(f"Update dataset.json → \"numTraining\": {total_train}")
        print("Then run: nnUNetv2_plan_and_preprocess -d 1 -c 2d -np 4")


if __name__ == "__main__":
    main()
