#!/usr/bin/env python3
"""
Compute the Nyul standard scale from training images and save it so that
NyulNormalization can use it during preprocessing and inference.

Output: {nnUNet_preprocessed}/{dataset}/nyul_standard_scale.json

Usage
-----
    python compute_nyul_scale.py
    python compute_nyul_scale.py --dataset Dataset001_NerveMAVI
    python compute_nyul_scale.py --landmarks 1 10 20 30 40 50 60 70 80 90 99

The script reads nnUNet_raw and nnUNet_preprocessed from environment
variables (falls back to standard paths under the workspace root).
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image


# ── Default paths (same as startEnvironment.sh) ────────────────────────────
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RAW          = os.path.join(_WORKSPACE, 'nnUNet_raw')
_DEFAULT_PREPROCESSED = os.path.join(_WORKSPACE, 'nnUNet_preprocessed')

# ── Landmark percentiles ────────────────────────────────────────────────────
DEFAULT_LANDMARKS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def compute_nyul_scale(images_dir: str, landmarks: list, output_path: str) -> None:
    img_files = sorted(glob.glob(os.path.join(images_dir, '*.tif')))

    # Each physical case has one file per channel (_0000, _0001, …).
    # Keep only the first channel per case for grayscale datasets.
    _ch_re = re.compile(r'_\d{4}$')
    seen, unique_files = set(), []
    for f in img_files:
        case = _ch_re.sub('', os.path.splitext(os.path.basename(f))[0])
        if case not in seen:
            seen.add(case)
            unique_files.append(f)

    if not unique_files:
        raise FileNotFoundError(f'No .tif files found in {images_dir}')

    print(f'Computing Nyul standard scale from {len(unique_files)} training images')
    print(f'Landmark percentiles: {landmarks}')

    all_landmarks = []
    for idx, path in enumerate(unique_files):
        img = np.array(Image.open(path)).astype(np.float64)
        lms = np.percentile(img, landmarks)
        all_landmarks.append(lms)
        if (idx + 1) % 200 == 0 or (idx + 1) == len(unique_files):
            print(f'  {idx + 1}/{len(unique_files)} images processed')

    all_landmarks = np.array(all_landmarks)          # shape: (N, L)
    standard_scale = all_landmarks.mean(axis=0)      # shape: (L,)

    # Normalise to [0, 1] so NyulNormalization outputs are always in [0, 1]
    s_min = standard_scale[0]
    s_max = standard_scale[-1]
    if s_max > s_min:
        standard_scale = (standard_scale - s_min) / (s_max - s_min)
    else:
        standard_scale = np.linspace(0.0, 1.0, len(landmarks))

    result = {
        'percentiles':     list(map(float, landmarks)),
        'standard_scale':  standard_scale.tolist(),
        'n_images':        len(unique_files),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nSaved → {output_path}')
    print('Standard scale (normalised to [0,1]):')
    for p, v in zip(landmarks, standard_scale):
        print(f'  p{p:>3}  →  {v:.5f}')
    print('\nDone.  You can now use channel_name "nyul" in dataset.json and')
    print('re-run nnUNetv2_plan_and_preprocess to activate NyulNormalization.')


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset',   default='Dataset001_NerveMAVI',
                        help='Dataset folder name inside nnUNet_raw (default: Dataset001_NerveMAVI)')
    parser.add_argument('--landmarks', nargs='+', type=float,
                        default=DEFAULT_LANDMARKS,
                        help='Percentile landmarks (default: 1 10 20 30 40 50 60 70 80 90 99)')
    args = parser.parse_args()

    nnunet_raw          = os.environ.get('nnUNet_raw',          _DEFAULT_RAW)
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed', _DEFAULT_PREPROCESSED)

    images_dir  = os.path.join(nnunet_raw,          args.dataset, 'imagesTr')
    output_path = os.path.join(nnunet_preprocessed, args.dataset, 'nyul_standard_scale.json')

    compute_nyul_scale(images_dir, args.landmarks, output_path)
