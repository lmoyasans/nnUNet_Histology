from abc import ABC, abstractmethod
from typing import Type
import os

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image -= mean
            image /= (max(std, 1e-8))
        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype, copy=False)
        image /= 255.
        return image


class HistologyNormalization(ImageNormalization):
    """
    Normalization for grayscale histology images.

    Clips intensity to the [low_percentile, high_percentile] range computed
    per-image, then rescales linearly to [0, 1].  This removes staining
    outliers and scanner-dependent intensity offsets without requiring
    dataset-level statistics.

    Activated automatically when the channel name in dataset.json is
    'histology' (case-insensitive).
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    # Percentile bounds – can be overridden by subclasses
    low_percentile: float = 1.0
    high_percentile: float = 99.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        p_low, p_high = np.percentile(image, [self.low_percentile, self.high_percentile])
        if p_high <= p_low:
            # Flat image – nothing to normalise; return zeros
            image -= p_low
            return image
        np.clip(image, p_low, p_high, out=image)
        image -= p_low
        image /= (p_high - p_low)   # → [0, 1]
        return image


class NyulNormalization(ImageNormalization):
    """
    Nyul & Udupa (1999) histogram landmark-based intensity standardisation.

    Requires a pre-computed standard scale file that encodes the mean
    percentile landmark positions across all training images.  Compute it
    once with:

        python compute_nyul_scale.py

    This writes  {nnUNet_preprocessed}/{dataset}/nyul_standard_scale.json.

    Algorithm
    ---------
    Training phase (compute_nyul_scale.py):
      1. For each training image compute percentile values at LANDMARKS.
      2. Average across all images → "standard scale".
      3. Normalise standard scale to [0, 1].

    Inference phase (this class):
      1. Compute this image's percentile values at the same LANDMARKS.
      2. Piecewise-linearly interpolate so those values map to the standard
         scale → output in [0, 1].

    Falls back to per-image HistologyNormalization if the scale file is not
    found (with a one-time warning), so preprocessing still works before the
    scale is computed.

    Activated automatically when the channel name in dataset.json is
    'nyul' (case-insensitive).
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    # Percentile landmarks — must match what was used in compute_nyul_scale.py
    LANDMARKS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    # Class-level cache: file_path → (percentiles_array, standard_scale_array)
    # Loaded once per process, shared across all instances.
    _scale_cache: dict = {}
    _warned: bool = False

    @classmethod
    def _find_scale_file(cls) -> str:
        """Locate nyul_standard_scale.json under nnUNet_preprocessed."""
        import glob as _glob
        preprocessed_root = os.environ.get('nnUNet_preprocessed', '')
        if not preprocessed_root:
            raise RuntimeError(
                'nnUNet_preprocessed env var not set. '
                'Cannot locate nyul_standard_scale.json.'
            )
        matches = _glob.glob(
            os.path.join(preprocessed_root, '*', 'nyul_standard_scale.json')
        )
        if not matches:
            raise FileNotFoundError(
                f'nyul_standard_scale.json not found under {preprocessed_root}. '
                'Run: python compute_nyul_scale.py'
            )
        return matches[0]

    @classmethod
    def _load_scale(cls):
        """Return (percentiles, standard_scale) arrays, cached per file path."""
        import json
        path = cls._find_scale_file()
        if path not in cls._scale_cache:
            with open(path) as f:
                data = json.load(f)
            cls._scale_cache[path] = (
                np.array(data['percentiles'], dtype=np.float64),
                np.array(data['standard_scale'], dtype=np.float64),
            )
        return cls._scale_cache[path]

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)

        try:
            percentiles, standard_scale = self._load_scale()
        except (RuntimeError, FileNotFoundError) as e:
            if not NyulNormalization._warned:
                import warnings
                warnings.warn(
                    f'NyulNormalization: {e}  '
                    'Falling back to per-image HistologyNormalization.',
                    RuntimeWarning, stacklevel=2
                )
                NyulNormalization._warned = True
            # Fallback: HistologyNormalization behaviour
            p_low, p_high = np.percentile(image, [1.0, 99.0])
            if p_high > p_low:
                np.clip(image, p_low, p_high, out=image)
                image -= p_low
                image /= (p_high - p_low)
            return image

        # Compute per-image landmark values
        img_landmarks = np.percentile(image, percentiles)

        # Piecewise-linear warp: map this image's landmarks → standard scale
        image_out = np.interp(image.ravel(), img_landmarks, standard_scale)
        image_out = image_out.reshape(image.shape).astype(self.target_dtype, copy=False)

        # Clip to [0, 1] (standard_scale is normalised to this range)
        np.clip(image_out, 0.0, 1.0, out=image_out)
        return image_out

