"""
ImageEnh.py
-----------
Fuzzy-based enhancement for GRAYSCALE images, using local statistics
and a Gaussian fuzzy membership function.

Pipeline (mirrors the project specification):
  1. Normalize pixel values from [0, 255] to [-1, +1]
  2. Divide the image into overlapping local fuzzy windows (multi-scale)
  3. Compute each pixel's fuzzy membership value with respect to each window
  4. Compute local mean / variance / std for each window
  5. Compute pixel_difference = pixel - local_mean
  6. Normalize the difference by the local standard deviation
  7. Weight the normalized difference by the fuzzy membership value
  8. Combine contributions from all relevant fuzzy windows
  9. Generate the enhanced pixel value
  10. Convert back from [-1, +1] to [0, 255]
"""

import time
import numpy as np

from utils import normalize, denormalize, local_statistics, fuzzy_membership, window_plan, EPS


def enhance_grayscale(
    image: np.ndarray,
    window_size: int = 9,
    gamma: float = 0.45,
    strength: float = 1.20,
    num_scales: int = 3,
):
    """Apply fuzzy-based local-statistics enhancement to a single-channel
    (grayscale) image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W), dtype uint8
        Input grayscale image, pixel values in [0, 255].
    window_size : int
        Base size of the local fuzzy window (odd, in pixels).
    gamma : float
        Spread of the Gaussian fuzzy membership function.
    strength : float
        Overall enhancement gain.
    num_scales : int
        Number of overlapping window scales to fuse (1-3).

    Returns
    -------
    enhanced : np.ndarray, shape (H, W), dtype uint8
    meta : dict with processing_time_ms and the parameters used.
    """
    start = time.perf_counter()

    norm = normalize(image)  # step 1
    windows = window_plan(window_size, num_scales)  # step 2

    combined = np.zeros_like(norm)
    for size, weight in windows:
        mean, variance, std = local_statistics(norm, size)  # step 4 (local stats)
        diff = norm - mean  # step 5
        normalized_diff = diff / (std + EPS)  # step 6
        mu = fuzzy_membership(normalized_diff, gamma)  # step 3
        combined += weight * mu * normalized_diff * strength  # steps 7 & 8

    enhanced_norm = np.clip(norm + combined, -1.0, 1.0)  # step 9
    enhanced = denormalize(enhanced_norm)  # step 10

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    meta = {
        "processing_time_ms": round(elapsed_ms, 2),
        "window_size": window_size,
        "gamma": gamma,
        "strength": strength,
        "num_scales": num_scales,
        "channels": 1,
    }
    return enhanced, meta


def image_statistics(image: np.ndarray) -> dict:
    """Basic quality statistics used for the before/after comparison cards."""
    img = image.astype(np.float64)
    return {
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "contrast": float(img.max() - img.min()),
    }
