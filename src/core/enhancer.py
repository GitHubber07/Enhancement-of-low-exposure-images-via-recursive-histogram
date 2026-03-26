import cv2
import numpy as np

# Set L to 256 for 8-bit images as per the original paper
L = 256


def calculate_exposure(histogram: np.ndarray) -> float:
    """
    Calculates the normalized exposure value of an image.

    Args:
        histogram (np.ndarray): The 1D histogram array of the image.

    Returns:
        float: The exposure value between 0 and 1.
    """
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0.0

    # exposure = sum(h(k)*k) / (L * sum(h(k)))
    weighted_sum = np.sum([k * histogram[k] for k in range(L)])
    exposure = weighted_sum / (L * total_pixels)
    return exposure[0] if isinstance(exposure, np.ndarray) else exposure


def apply_esihe(img_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Applies a single iteration of Exposure-based Sub-image Histogram Equalization.

    Args:
        img_gray (np.ndarray): The input grayscale or single-channel image.

    Returns:
        tuple: A tuple containing the enhanced single-channel image and the exposure value.
    """
    # Step 1: Compute histogram
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()

    # Step 2: Compute exposure and X_a threshold
    exposure = calculate_exposure(hist)
    Xa = int(L * (1 - exposure))

    # Ensure Xa is within sensible bounds to prevent division by zero or errors
    Xa = max(1, min(Xa, L - 2))

    # Step 3: Histogram clipping
    Tc = np.sum(hist) / L
    hist_clipped = np.clip(hist, 0, Tc)

    # Step 4 & 5: Sub-divide and calculate CDFs
    hist_L = hist_clipped[0:Xa]
    hist_U = hist_clipped[Xa:L]

    N_L = np.sum(hist_L)
    N_U = np.sum(hist_U)

    # Calculate PDFs
    P_L = hist_L / N_L if N_L > 0 else np.zeros_like(hist_L)
    P_U = hist_U / N_U if N_U > 0 else np.zeros_like(hist_U)

    # Calculate CDFs
    C_L = np.cumsum(P_L)
    C_U = np.cumsum(P_U)

    # Calculate transfer functions
    F_L = np.asarray(Xa * C_L, dtype=np.float32)
    F_U = np.asarray((Xa + 1) + (L - Xa - 1) * C_U, dtype=np.float32)

    # Step 6: Map pixels to new values
    img_mapped = np.zeros_like(img_gray, dtype=np.float32)

    # Apply mapping using logical lower/upper masks
    lower_mask = img_gray < Xa
    upper_mask = img_gray >= Xa

    img_mapped[lower_mask] = F_L[img_gray[lower_mask]]
    img_mapped[upper_mask] = F_U[img_gray[upper_mask] - Xa]

    return img_mapped.astype(np.uint8), float(exposure)


def apply_r_esihe(img_gray: np.ndarray, epsilon: float = 0.01, max_iter: int = 10) -> tuple[np.ndarray, int]:
    """
    Recursively applies ESIHE until the exposure difference stabilizes below epsilon.

    Args:
        img_gray (np.ndarray): Grayscale or Luma channel input.
        epsilon (float): Threshold to decide if convergence is met.
        max_iter (int): Maximum bounding iterations to prevent infinite loops.

    Returns:
        tuple: A tuple containing the finalized image and number of iterations completed.
    """
    current_img = img_gray.copy()
    prev_exposure = -1.0

    num_iterations = 0
    for i in range(max_iter):
        enhanced_img, current_exposure = apply_esihe(current_img)
        num_iterations += 1

        # Stop condition: The exposure stabilized (difference < epsilon)
        if prev_exposure != -1.0 and abs(current_exposure - prev_exposure) < epsilon:
            break

        current_img = enhanced_img
        prev_exposure = current_exposure

    return current_img, num_iterations


def process_color_image(img_bgr: np.ndarray, epsilon: float = 0.01, max_iter: int = 10) -> tuple[np.ndarray, int]:
    """
    Applies R-ESIHE safely on a BGR color image to preserve color saturation and hue.
    It applies the algorithm exclusively to the Value channel in HSV color space.

    Args:
        img_bgr (np.ndarray): The BGR image.
        epsilon (float): Convergence threshold.
        max_iter (int): Maximum iterations.

    Returns:
        tuple: (Enhanced BGR image, number of iterations used)
    """
    # Convert to HSV color space to process intensity separately
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply Recursive ESIHE to the Intensity/Value channel
    v_enhanced, iters = apply_r_esihe(v, epsilon=epsilon, max_iter=max_iter)

    # Merge channels and return to BGR space
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    img_enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return img_enhanced_bgr, iters
