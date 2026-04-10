"""
prepare_extra_channels.py
=========================
Generate extra input channels from the existing _0000.tif images:

  _0000.tif  –  UNTOUCHED (original raw image)
  _0001.tif  –  Laplacian of Gaussian (LoG) blob response (float32, [0,1])
  _0002.tif  –  Local dark contrast map (float32, [0,1])

Features are designed to detect DARK CIRCULAR FASCICLES:
- LoG responds strongly to dark blob-like structures (nerve fascicles)
- Local dark contrast highlights pixels darker than their neighborhood

Usage
-----
    python prepare_extra_channels.py [--workers 8] [--dataset Dataset001_NerveMAVI]

The script skips files that already exist unless --force is passed.
"""

import argparse
import glob
import multiprocessing as mp
import os

import numpy as np
import tifffile
from scipy.ndimage import gaussian_laplace, uniform_filter


# ── Per-image worker ──────────────────────────────────────────────────────────

def process_image(src_path: str, force: bool) -> str:
    """
    Given  …/CASE_0000.tif  produce:
       …/CASE_0001.tif   (LoG blob response, float32)
       …/CASE_0002.tif   (Local dark contrast, float32)

    The original _0000.tif is NEVER modified.
    Returns a short status string for logging.
    """
    base = src_path[:-len("_0000.tif")]  # strip suffix
    dst_log     = base + "_0001.tif"
    dst_dark    = base + "_0002.tif"

    if not force and os.path.exists(dst_log) and os.path.exists(dst_dark):
        return f"SKIP  {os.path.basename(src_path)}"

    # Load RAW image
    img_raw = tifffile.imread(src_path)
    
    # Normalize to [0,1] for processing
    img = img_raw.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0  # assume uint8 input

    # --- Channel 1: Laplacian of Gaussian (LoG) for dark blob detection -----
    #   LoG is positive inside dark blobs (fascicles) and negative on edges.
    #   We use multiple scales to capture fascicles of different sizes.
    #   Fascicles can range from ~20 to ~200+ pixels in diameter, so we use
    #   sigmas roughly 1/3 of the expected radii.
    sigmas = [5, 10, 20, 40, 80]  # covers ~15-240 pixel diameter blobs
    
    # Compute scale-normalized LoG at each sigma, take maximum response
    log_responses = []
    for sigma in sigmas:
        # gaussian_laplace returns negative inside bright blobs, positive inside dark
        log = -gaussian_laplace(img, sigma=sigma) * (sigma ** 2)  # scale normalization
        log_responses.append(log)
    
    # Take maximum positive response across scales (dark blob detector)
    log_max = np.maximum.reduce(log_responses)
    log_max = np.clip(log_max, 0, None)  # keep only positive (dark blob) responses
    
    # Normalize to [0, 1]
    l_max = log_max.max()
    if l_max > 0:
        log_max /= l_max
    log_max = log_max.astype(np.float32)

    # --- Channel 2: Local dark contrast map ----------------------------------
    #   Highlights pixels that are darker than their local neighborhood.
    #   This helps identify fascicle interiors which are darker than surrounding
    #   perineurium and connective tissue.
    #   
    #   dark_contrast = max(0, local_mean - pixel_value) / local_range
    
    # Use multiple window sizes and combine
    window_sizes = [31, 61, 121]  # small, medium, large neighborhoods
    dark_maps = []
    
    for win in window_sizes:
        local_mean = uniform_filter(img, size=win, mode='reflect')
        local_min = uniform_filter(img, size=win, mode='reflect', output=np.float32)
        # For local max/min, we use a trick with -img
        # Actually, let's use a simpler approach: local_mean - pixel
        dark_diff = local_mean - img  # positive where pixel is darker than surroundings
        dark_diff = np.clip(dark_diff, 0, None)  # keep only "darker than local" pixels
        dark_maps.append(dark_diff)
    
    # Combine: take maximum response across window sizes
    dark_contrast = np.maximum.reduce(dark_maps)
    
    # Normalize to [0, 1]
    d_max = dark_contrast.max()
    if d_max > 0:
        dark_contrast /= d_max
    dark_contrast = dark_contrast.astype(np.float32)

    # Write channels (original _0000.tif is untouched)
    tifffile.imwrite(dst_log,  log_max,       photometric="minisblack")
    tifffile.imwrite(dst_dark, dark_contrast, photometric="minisblack")

    return f"OK    {os.path.basename(src_path)}"

    return f"OK    {os.path.basename(src_path)}"


# ── Pool initializer (share settings without pickling each call) ──────────────

_force: bool = False


def _init_worker(force):
    global _force
    _force = force


def _worker(src_path: str) -> str:
    return process_image(src_path, _force)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Generate gradient + Frangi extra channels for nnUNet.")
    parser.add_argument("--dataset",  default="Dataset001_NerveMAVI")
    parser.add_argument("--workers",  type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--force",    action="store_true", help="Overwrite existing _0001/_0002 files")
    args = parser.parse_args()

    raw_dir = os.path.join(workspace, "nnUNet_raw", args.dataset)

    print(f"Dataset            : {args.dataset}")
    print(f"Sobel/Frangi on    : RAW image (original _0000.tif is NEVER modified)")

    # Collect all _0000.tif images (training + test)
    all_srcs = sorted(
        glob.glob(os.path.join(raw_dir, "imagesTr", "*_0000.tif")) +
        glob.glob(os.path.join(raw_dir, "imagesTs", "*_0000.tif"))
    )
    print(f"Images to process  : {len(all_srcs)}  (workers={args.workers})")

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.force,),
    ) as pool:
        for i, msg in enumerate(pool.imap_unordered(_worker, all_srcs, chunksize=32), 1):
            if i % 500 == 0 or i == len(all_srcs):
                print(f"  [{i:>5}/{len(all_srcs)}]  {msg}")

    print("\nDone — extra channels written alongside _0000.tif files.")
    print("Next steps:")
    print("  1. Update dataset.json channel_names to include channels 1 and 2")
    print("  2. Run: nnUNetv2_plan_and_preprocess -d 1 -c 2d -overwrite_plans_name nnUNetPlansMultiCh -np 4")
    print("  3. Train: nnUNetv2_train 1 2d 0 -tr nnUNetTrainerAdamEarlyStopping_LowLR -p nnUNetPlansMultiCh")


if __name__ == "__main__":
    main()
