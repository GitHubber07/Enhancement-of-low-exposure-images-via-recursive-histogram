import math
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(target: np.ndarray, output: np.ndarray) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between a ground-truth and output image.

    Args:
        target (np.ndarray): Ground truth reference image.
        output (np.ndarray): Enhanced/processed output image.

    Returns:
        float: PSNR value.
    """
    mse = np.mean((target.astype(np.float64) - output.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0  # Perfect match
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(target: np.ndarray, output: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index (SSIM) between target and output.

    Args:
        target (np.ndarray): Ground truth reference image.
        output (np.ndarray): Enhanced/processed output image.

    Returns:
        float: SSIM value.
    """
    # channel_axis=2 works correctly with HWC format (e.g. RGB/BGR)
    # Check if grayscale or multi-channel
    if len(target.shape) == 3:
        return ssim(target, output, channel_axis=2)
    else:
        return ssim(target, output)


def calculate_metrics(target: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    """
    Calculates PSNR and SSIM.

    Args:
        target (np.ndarray): Reference image.
        output (np.ndarray): Processed image.

    Returns:
        tuple: (PSNR, SSIM)
    """
    return calculate_psnr(target, output), calculate_ssim(target, output)
