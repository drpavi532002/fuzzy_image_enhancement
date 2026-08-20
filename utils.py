"""
utils.py
--------
Shared mathematical / helper operations for the fuzzy-based image
enhancement algorithm: normalization, local statistics (mean, variance,
standard deviation) over sliding windows, and the fuzzy membership
function. Both ImageEnh.py (grayscale) and ColoredImageEnh.py (RGB)
build on top of these primitives so the core math lives in one place.
"""

import numpy as np

EPS = 1e-6  # avoids divide-by-zero in flat (zero-variance) regions


def normalize(image_0_255: np.ndarray) -> np.ndarray:
    """Rescale pixel intensities from [0, 255] to [-1, +1].

    x_norm = 2 * (x / 255) - 1
    """
    return 2.0 * (image_0_255.astype(np.float64) / 255.0) - 1.0


def denormalize(image_norm: np.ndarray) -> np.ndarray:
    """Rescale pixel intensities from [-1, +1] back to [0, 255] (uint8)."""
    out = (image_norm + 1.0) / 2.0 * 255.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def local_statistics(norm_image: np.ndarray, window_size: int):
    """Compute the local mean, variance and standard deviation for every
    pixel using a `window_size` x `window_size` sliding (fuzzy) window,
    via a box filter implemented with an integral image for O(1) lookups
    per pixel regardless of window size.

    Returns
    -------
    mean, variance, std : np.ndarray
        Same shape as `norm_image`.
    """
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    h, w = norm_image.shape
    # Pad with edge replication so windows near the border are well defined.
    padded = np.pad(norm_image, half, mode="edge")
    padded_sq = padded ** 2

    # Integral images (summed-area tables)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral_sq = padded_sq.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    integral_sq = np.pad(integral_sq, ((1, 0), (1, 0)), mode="constant")

    win = window_size
    # Box sums via the standard integral-image trick, vectorized over the image.
    total = (
        integral[win:, win:] - integral[:-win, win:]
        - integral[win:, :-win] + integral[:-win, :-win]
    )
    total_sq = (
        integral_sq[win:, win:] - integral_sq[:-win, win:]
        - integral_sq[win:, :-win] + integral_sq[:-win, :-win]
    )
    n = win * win
    mean = total / n
    variance = np.clip(total_sq / n - mean ** 2, a_min=0, a_max=None)
    std = np.sqrt(variance)

    # integral-image box filter output is already aligned to the original image
    return mean[:h, :w], variance[:h, :w], std[:h, :w]


def fuzzy_membership(normalized_difference: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian fuzzy membership function.

    membership(x) = exp( -x^2 / (2 * gamma^2) )

    A pixel that sits close to its local mean (small normalized
    difference) gets low membership ("typical" -> little enhancement);
    a pixel far from its local mean (an edge, texture, noise) gets a
    membership closer to 1 ("unusual" -> stronger enhancement).
    """
    gamma = max(gamma, EPS)
    return np.exp(-(normalized_difference ** 2) / (2 * gamma ** 2))


def window_plan(base_size: int, num_scales: int = 3):
    """Build the set of overlapping fuzzy window sizes and their fusion
    weights used to combine contributions from multiple windows, as
    described in the project spec (step 9: combine contributions from
    all relevant fuzzy windows).
    """
    raw_sizes = [base_size, round(base_size * 1.6), round(base_size * 2.2)]
    sizes = []
    for s in raw_sizes[:max(1, min(3, num_scales))]:
        sizes.append(s if s % 2 == 1 else s + 1)
    raw_weights = [0.5, 0.3, 0.2][:len(sizes)]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]
    return list(zip(sizes, weights))
