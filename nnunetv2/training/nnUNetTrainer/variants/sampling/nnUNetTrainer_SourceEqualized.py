"""
nnUNetTrainer_SourceEqualized.py
=================================
Per-source-image equalized oversampling + optional histology-aware augmentation.

Two orthogonal mixins address the Bio-Aegis / GTEx domain gap:

  _SourceEqualizedOversampleMixin
      Fixes *sampling frequency*: sources with few tiles (Bio-Aegis: 20 tiles)
      are duplicated up to max_factor=20× so every source image gets roughly
      equal expected draws per epoch.  Fully automatic — reads tile_index.json,
      no dataset-specific strings required.

  _HistologyAugMixin
      Fixes *appearance generalization*: widens the intensity, blur, and contrast
      augmentation ranges beyond nnUNet's defaults so the model learns features
      that are invariant to staining darkness, scanner contrast, and blur level.
      This is the complementary fix to oversampling:
        • Oversampling = see Bio-Aegis tiles more often
        • Wider augmentation = learn features that generalize across stain domains

      Concretely it wraps the parent get_training_transforms() result and
      appends stronger versions of the transforms already present:
        • MultiplicativeBrightness: (0.65, 1.35) at p=0.30  [default (0.75,1.25) p=0.15]
        • Contrast:                 (0.65, 1.35) at p=0.30  [default (0.75,1.25) p=0.15]
        • GaussianBlur:             (0.5, 2.0)   at p=0.30  [default (0.5, 1.0)  p=0.20]
        • GaussianNoise:            (0, 0.15)    at p=0.20  [default (0, 0.1)    p=0.10]
      Inserted just before the DownsampleSegForDSTransform so deep supervision
      targets are not affected.

Concrete trainer classes
------------------------
  nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized
      TV_Nyul + equalized sampling  [recommended first Phase 2 run]

  nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized_HistoAug
      TV_Nyul + equalized sampling + wider histology augmentation  [recommended]

  nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft_SourceEqualized
      Softened focal Tversky + equalized sampling

  nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft_SourceEqualized_HistoAug
      Softened focal Tversky + equalized sampling + wider augmentation

Usage
-----
  nnUNetv2_train 1 2d 0 \\
      -tr nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized_HistoAug \\
      -p  nnUNetPlansNyulNorm

No configuration needed.  Add new datasets via the tiling scripts and
tile_index.json updates automatically.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdamEarlyStopping_Tversky import (
    nnUNetTrainerAdamEarlyStopping_Tversky,
    nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft,
)


# ── Mixin 1: source-equalized oversampling ───────────────────────────────────

class _SourceEqualizedOversampleMixin:
    """
    Equalizes training-tile sampling across source images.

    Every source image (unique value of tile_index[case]['source_file']) ends
    up with the same expected draw count per epoch, so no single slide dominates
    and no small-tile-count dataset is starved.

    Override `max_factor` (default 20) to cap maximum duplication per source.
    """

    max_factor: int = 20

    def _find_tile_index(self) -> dict:
        """Locate tile_index.json from the preprocessed dataset folder."""
        prep_root = Path(self.preprocessed_dataset_folder)
        dataset_name = prep_root.parent.name

        if nnUNet_raw is not None:
            candidate = Path(nnUNet_raw) / dataset_name / "tile_index.json"
            if candidate.exists():
                return json.loads(candidate.read_text())

        for parent in prep_root.parents:
            candidate = parent / "nnUNet_raw" / dataset_name / "tile_index.json"
            if candidate.exists():
                return json.loads(candidate.read_text())

        self.print_to_log_file(
            "[SourceEqualized] WARNING: tile_index.json not found — "
            "source-equalized oversampling disabled (falling back to default)."
        )
        return {}

    def _equalize_identifiers(self, tr_keys: list, tile_index: dict) -> list:
        """
        Duplicate underrepresented sources so all reach the pool size of the
        largest source, capped at max_factor per source.
        """
        source_to_keys: dict = defaultdict(list)
        unknown = []
        for k in tr_keys:
            src = tile_index.get(k, {}).get("source_file", "")
            if src:
                source_to_keys[src].append(k)
            else:
                unknown.append(k)

        if not source_to_keys:
            return tr_keys

        counts = {src: len(keys) for src, keys in source_to_keys.items()}
        max_count = max(counts.values())

        equalized = list(unknown)
        log_lines = ["[SourceEqualized] Per-source tile counts and duplication:"]

        for src, keys in sorted(source_to_keys.items()):
            n = len(keys)
            factor = min(math.ceil(max_count / n), self.max_factor)
            target = n * factor
            repeated = (keys * factor)[:target]
            equalized.extend(repeated)
            log_lines.append(
                f"  {Path(src).name[:60]:60s}  "
                f"{n:5d} tiles  ×{factor:2d}  → {target:5d}"
            )

        log_lines.append(
            f"  Total identifiers: {len(tr_keys)} → {len(equalized)} "
            f"({len(source_to_keys)} unique sources equalized)"
        )
        self.print_to_log_file("\n".join(log_lines))
        return sorted(equalized)

    def get_tr_and_val_datasets(self):
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()

        tile_index = self._find_tile_index()
        if not tile_index:
            return dataset_tr, dataset_val

        original_ids = list(dataset_tr.identifiers)
        equalized_ids = self._equalize_identifiers(original_ids, tile_index)
        dataset_tr.identifiers = equalized_ids

        return dataset_tr, dataset_val


# ── Mixin 2: histology-aware intensity augmentation ──────────────────────────

class _HistologyAugMixin:
    """
    Widens intensity/blur augmentation for histology domain generalization.

    nnUNet's default ranges are tuned for isotropic 3D medical volumes where
    staining is consistent.  For H&E histology across different scanners and
    staining protocols, wider ranges are needed to prevent the model from
    relying on absolute intensity or specific blur levels.

    This mixin overrides get_training_transforms() to add stronger versions
    of the intensity transforms on top of the base pipeline.  Insert position:
    just before DownsampleSegForDSTransform so deep supervision targets are
    unaffected.

    Added transforms (applied on top of defaults, not replacing them):
      MultiplicativeBrightness  (0.65, 1.35)  p=0.30   [default (0.75,1.25) p=0.15]
      Contrast                  (0.65, 1.35)  p=0.30   [default (0.75,1.25) p=0.15]
      GaussianBlur sigma        (0.5,  2.0 )  p=0.30   [default (0.5, 1.0)  p=0.20]
      GaussianNoise var         (0,    0.15)  p=0.20   [default (0, 0.1)    p=0.10]
    """

    def get_training_transforms(self, *args, **kwargs):
        # Get the base ComposeTransforms from the parent chain
        composed = super().get_training_transforms(*args, **kwargs)

        # Locate insertion point: just before DownsampleSegForDSTransform
        # (which must stay last so deep supervision targets are correct)
        t_list = composed.transforms
        ds_idx = next(
            (i for i, t in enumerate(t_list) if isinstance(t, DownsampleSegForDSTransform)),
            len(t_list),  # append at end if deep supervision not used
        )

        extra = [
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.65, 1.35)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.30,
            ),
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.65, 1.35)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.30,
            ),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 2.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.30,
            ),
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.15),
                    p_per_channel=1,
                    synchronize_channels=True,
                ),
                apply_probability=0.20,
            ),
        ]

        # Insert before DownsampleSegForDSTransform
        for i, t in enumerate(extra):
            t_list.insert(ds_idx + i, t)

        self.print_to_log_file(
            "[HistologyAug] Inserted 4 additional augmentation transforms "
            f"at position {ds_idx} (before DownsampleSegForDS):\n"
            "  MultiplicativeBrightness (0.65–1.35) p=0.30\n"
            "  Contrast                 (0.65–1.35) p=0.30\n"
            "  GaussianBlur sigma       (0.5–2.0)   p=0.30\n"
            "  GaussianNoise var        (0–0.15)    p=0.20"
        )
        return composed


# ── Concrete trainer classes ──────────────────────────────────────────────────

class nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized(
    _SourceEqualizedOversampleMixin,
    nnUNetTrainerAdamEarlyStopping_Tversky,
):
    """
    TV_Nyul trainer + automatic per-source-image equalized oversampling.

    Recommended first Phase 2 experiment.  Loss identical to TV_Nyul;
    only sampling distribution changes.  Sources with few tiles duplicated
    up to max_factor=20× to match the largest source.

    Plan: nnUNetPlansNyulNorm
    """
    max_factor = 20

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized_HistoAug(
    _HistologyAugMixin,
    _SourceEqualizedOversampleMixin,
    nnUNetTrainerAdamEarlyStopping_Tversky,
):
    """
    TV_Nyul + per-source equalized sampling + wider histology augmentation.

    Addresses the Bio-Aegis domain gap from both angles:
      • Sampling: sources with few tiles drawn proportionally more often
      • Augmentation: wider intensity/blur ranges → stain-invariant features

    This is the most complete Phase 2 experiment.  Run alongside
    SourceEqualized (no aug) to isolate the augmentation contribution.

    Plan: nnUNetPlansNyulNorm
    """
    max_factor = 20

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft_SourceEqualized(
    _SourceEqualizedOversampleMixin,
    nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft,
):
    """
    Softened per-class focal Tversky + automatic per-source-image equalization.

    Run after nnUNetTrainerAdamEarlyStopping_Tversky_SourceEqualized to
    determine whether the extra loss bias adds value on top of sampling alone.

    Plan: nnUNetPlansNyulNorm
    """
    max_factor = 20

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft_SourceEqualized_HistoAug(
    _HistologyAugMixin,
    _SourceEqualizedOversampleMixin,
    nnUNetTrainerAdamEarlyStopping_TverskyPerClassSoft,
):
    """
    Softened focal Tversky + equalized sampling + wider histology augmentation.

    All three fixes combined.  Run after the individual experiments to confirm
    that each component adds independent value.

    Plan: nnUNetPlansNyulNorm
    """
    max_factor = 20

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        self.initial_lr = 3e-4
        super().__init__(plans, configuration, fold, dataset_json, device)

