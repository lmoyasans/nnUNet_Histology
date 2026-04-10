"""
postprocess_predictions.py
==========================
Morphological post-processing of reconstructed full-image predictions.

Applied per non-background class (in priority order):
  1. Binary closing  — bridges narrow gaps / connects broken regions
                       (dilation then erosion with a disk SE)
  2. Hole filling    — fills completely enclosed background holes inside
                       a class region (scipy.ndimage.binary_fill_holes)

A class priority list resolves any pixel-level conflicts that arise after
closing (higher-priority class overwrites lower-priority):
  NerveFascicle (3) > Adipose (2) > ConnectivePerineurium (1) > Background (0)
  Blood_vessel (4) has highest priority — never overwritten.

Output:
  <reconstructed_dir>/<stem>_pred_postpro.tif   ← post-processed label map

Usage
-----
  # Process all 6 default experiments:
  python postprocess_predictions.py

  # Process a specific experiment:
  python postprocess_predictions.py --label CE_Histo

  # Custom closing radius (pixels), disable hole-fill:
  python postprocess_predictions.py --close-radius 8 --no-fill-holes

  # Process a specific pred file directly:
  python postprocess_predictions.py --input /path/to/pred.tif --output /path/to/out.tif
"""

import argparse
import os

import numpy as np
import tifffile
from PIL import Image

try:
    from scipy.ndimage import binary_fill_holes, binary_closing
    from scipy.ndimage import generate_binary_structure, iterate_structure
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ── Defaults ──────────────────────────────────────────────────────────────────
_WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_WORKSPACE, 'nnUNet_results', 'Dataset001_NerveMAVI')
FOLD         = 0

EXPERIMENTS = [
    ('CE_Histo',   'nnUNetTrainerAdamEarlyStopping_LowLR',   'nnUNetPlans'),
    ('CE_Nyul',    'nnUNetTrainerAdamEarlyStopping_LowLR',   'nnUNetPlansNyul'),
    ('CE_NoNorm',  'nnUNetTrainerAdamEarlyStopping_LowLR',   'nnUNetPlansNoNorm'),
    ('TV_Histo',   'nnUNetTrainerAdamEarlyStopping_Tversky', 'nnUNetPlans'),
    ('TV_Nyul',    'nnUNetTrainerAdamEarlyStopping_Tversky', 'nnUNetPlansNyul'),
    ('TV_NoNorm',  'nnUNetTrainerAdamEarlyStopping_Tversky', 'nnUNetPlansNoNorm'),
]

# Classes processed in ascending priority order (later = higher priority)
# Background (0) is implicit and never directly processed.
CLASS_PRIORITY = [1, 2, 3, 4]   # Connective, Adipose, NerveFascicle, Blood_vessel

DEFAULT_CLOSE_RADIUS = 6   # pixels — disk SE radius for binary closing
DEFAULT_FILL_HOLES   = True


# ── Morphological helpers ──────────────────────────────────────────────────────

def disk_se(radius: int) -> np.ndarray:
    """Create a 2-D disk structuring element of the given radius."""
    r  = int(radius)
    d  = 2 * r + 1
    se = np.zeros((d, d), dtype=bool)
    cy, cx = r, r
    for y in range(d):
        for x in range(d):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                se[y, x] = True
    return se


def postprocess_label_map(
        pred:          np.ndarray,
        close_radius:  int  = DEFAULT_CLOSE_RADIUS,
        fill_holes:    bool = DEFAULT_FILL_HOLES,
        class_priority: list = CLASS_PRIORITY,
) -> np.ndarray:
    """
    Apply morphological closing + hole filling to a reconstructed label map.

    Parameters
    ----------
    pred          : uint8 2-D label map (values = class indices)
    close_radius  : radius of the disk SE used for binary closing (pixels)
    fill_holes    : if True, fill enclosed background holes in each class
    class_priority: classes processed in ascending priority (last = highest)

    Returns
    -------
    postpro : uint8 2-D label map (same shape as pred)
    """
    if not _SCIPY:
        raise ImportError('scipy is required: pip install scipy')

    se      = disk_se(close_radius)
    postpro = pred.copy()

    for cls in class_priority:
        mask = (pred == cls)
        if not mask.any():
            continue

        # 1. Morphological closing (dilation → erosion)
        closed = binary_closing(mask, structure=se)

        # 2. Fill entirely enclosed holes
        if fill_holes:
            closed = binary_fill_holes(closed)

        # Write back — higher-priority classes overwrite lower ones
        # Pixels that were Background and are now claimed: assign to cls
        # Pixels already belonging to a *higher*-priority class: keep theirs
        # Strategy: write closed mask, then later higher-priority classes
        #           will overwrite if needed (loop is ascending priority)
        new_pixels        = closed & ~mask          # newly claimed pixels
        # Only overwrite if currently background or lower-priority class
        can_overwrite     = np.isin(postpro[new_pixels],
                                    [0] + [c for c in class_priority if c < cls])
        coords            = np.argwhere(new_pixels)
        valid             = coords[can_overwrite]
        if valid.size:
            postpro[valid[:, 0], valid[:, 1]] = cls

        # Fill holes: enclosed background should become this class
        # (only if the surrounding label is already this class in postpro)
        if fill_holes:
            filled           = binary_fill_holes(postpro == cls)
            hole_mask        = filled & (postpro == 0)
            postpro[hole_mask] = cls

    return postpro


def process_file(
        input_path:    str,
        output_path:   str,
        close_radius:  int,
        fill_holes:    bool,
        verbose:       bool = True,
) -> None:
    pred    = np.array(Image.open(input_path))
    postpro = postprocess_label_map(pred,
                                    close_radius=close_radius,
                                    fill_holes=fill_holes)
    tifffile.imwrite(output_path, postpro)
    if verbose:
        changed = int((pred != postpro).sum())
        total   = pred.size
        print(f'    {os.path.basename(input_path)}  →  {os.path.basename(output_path)}'
              f'   changed {changed:,} / {total:,} px  ({100*changed/total:.2f}%)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label',        default=None,
                        help='Process only this experiment label (e.g. CE_Histo). '
                             'Default: process all 6 experiments.')
    parser.add_argument('--input',        default=None,
                        help='Process a single pred .tif directly (overrides --label).')
    parser.add_argument('--output',       default=None,
                        help='Output path when --input is used.')
    parser.add_argument('--close-radius', type=int, default=DEFAULT_CLOSE_RADIUS,
                        help=f'Disk SE radius for morphological closing (px). '
                             f'Default: {DEFAULT_CLOSE_RADIUS}')
    parser.add_argument('--no-fill-holes', dest='fill_holes', action='store_false',
                        help='Disable binary hole filling.')
    parser.set_defaults(fill_holes=DEFAULT_FILL_HOLES)
    parser.add_argument('--fold', type=int, default=FOLD)
    args = parser.parse_args()

    if not _SCIPY:
        print('ERROR: scipy not found. Install with: pip install scipy')
        return

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.input:
        out = args.output or args.input.replace('_pred.tif', '_pred_postpro.tif')
        print(f'Processing single file: {args.input}')
        process_file(args.input, out, args.close_radius, args.fill_holes)
        return

    # ── Experiment mode ───────────────────────────────────────────────────────
    exps = [(l, t, p) for l, t, p in EXPERIMENTS
            if args.label is None or l == args.label]
    if not exps:
        print(f'No matching experiment for label="{args.label}"')
        return

    print(f'Close radius : {args.close_radius} px')
    print(f'Fill holes   : {args.fill_holes}')
    print()

    for label, trainer, plan in exps:
        recon_dir = os.path.join(RESULTS_ROOT,
                                 f'{trainer}__{plan}__2d',
                                 f'fold_{args.fold}', 'reconstructed')
        pred_files = sorted(f for f in os.listdir(recon_dir)
                            if f.endswith('_pred.tif')) if os.path.isdir(recon_dir) else []
        if not pred_files:
            print(f'[SKIP] {label} — no _pred.tif found in {recon_dir}')
            continue

        print(f'[{label}]  {len(pred_files)} images')
        for fname in pred_files:
            inp = os.path.join(recon_dir, fname)
            out = os.path.join(recon_dir, fname.replace('_pred.tif', '_pred_postpro.tif'))
            process_file(inp, out, args.close_radius, args.fill_holes)

    print('\nDone.')


if __name__ == '__main__':
    main()
