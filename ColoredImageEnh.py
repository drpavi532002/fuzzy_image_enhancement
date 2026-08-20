"""
ColoredImageEnh.py
-------------------
Applies the same fuzzy-based local-statistics enhancement used in
ImageEnh.py independently to the three color channels of an RGB image
(Red, Green, Blue), then recombines them into the final enhanced
color image.

    RGB Image
       |
       +--> R channel --> fuzzy enhancement --+
       +--> G channel --> fuzzy enhancement --+--> merge --> Enhanced RGB
       +--> B channel --> fuzzy enhancement --+
"""

import time
import numpy as np

from ImageEnh import enhance_grayscale, image_statistics


def enhance_color(
    image: np.ndarray,
    window_size: int = 9,
    gamma: float = 0.45,
    strength: float = 1.20,
    num_scales: int = 3,
):
    """Apply fuzzy-based enhancement independently to each RGB channel.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    enhanced : np.ndarray, shape (H, W, 3), dtype uint8
    meta : dict with processing_time_ms and per-channel info.
    """
    start = time.perf_counter()

    channels = []
    for c in range(3):  # R, G, B independently
        enhanced_channel, _ = enhance_grayscale(
            image[:, :, c],
            window_size=window_size,
            gamma=gamma,
            strength=strength,
            num_scales=num_scales,
        )
        channels.append(enhanced_channel)

    enhanced = np.stack(channels, axis=-1)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    meta = {
        "processing_time_ms": round(elapsed_ms, 2),
        "window_size": window_size,
        "gamma": gamma,
        "strength": strength,
        "num_scales": num_scales,
        "channels": 3,
    }
    return enhanced, meta


def is_grayscale(image: np.ndarray, tolerance: int = 6) -> bool:
    """Detect whether an RGB image is effectively grayscale (R≈G≈B)."""
    if image.ndim == 2:
        return True
    r, g, b = image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)
    return bool(
        np.max(np.abs(r - g)) <= tolerance
        and np.max(np.abs(g - b)) <= tolerance
        and np.max(np.abs(r - b)) <= tolerance
    )


def color_image_statistics(image: np.ndarray) -> dict:
    """Aggregate quality statistics across all three channels."""
    return image_statistics(image)
