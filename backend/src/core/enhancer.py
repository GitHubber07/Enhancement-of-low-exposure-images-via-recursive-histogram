import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import List

@dataclass
class TelemetryMetricsDto:
    algorithm_used: str
    original_exposure: float
    enhanced_exposure: float
    original_entropy: float
    enhanced_entropy: float

@dataclass
class TelemetryHistogramsDto:
    original: List[int]
    enhanced: List[int]

@dataclass
class PipelineResultDto:
    image_data: np.ndarray
    metrics: TelemetryMetricsDto
    histograms: TelemetryHistogramsDto

L = 256

def calculate_exposure(histogram: np.ndarray) -> float:
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0.0
    weighted_sum = np.sum([k * histogram[k] for k in range(len(histogram))])
    exposure = weighted_sum / (len(histogram) * total_pixels)
    return float(exposure)

def calculate_entropy(img_gray: np.ndarray) -> float:
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    hist_prob = hist / np.sum(hist)
    entropy = -np.sum([p * math.log2(p) for p in hist_prob if p > 0])
    return entropy

def get_adaptive_max_iter(img_gray: np.ndarray) -> int:
    entropy = calculate_entropy(img_gray)
    if entropy < 4.0:
        return 2  # Low complexity -> fewer iterations
    elif entropy < 6.0:
        return 4
    else:
        return 6  # High complexity -> more iterations

def process_sub_histogram(img_gray: np.ndarray, lower_bound: int, upper_bound: int, Xa: int, clip_limit: float) -> tuple[np.ndarray, np.ndarray]:
    # Ensure Xa is within bounds
    Xa = max(lower_bound + 1, min(Xa, upper_bound - 1))
    
    # Isolate relevant parts of the histogram
    mask = (img_gray >= lower_bound) & (img_gray < upper_bound)
    relevant_pixels = img_gray[mask]
    
    if len(relevant_pixels) == 0:
        return img_gray, img_gray.copy()
    
    hist = cv2.calcHist([img_gray[mask]], [0], None, [256], [0, 256]).flatten()
    
    hist_clipped = np.clip(hist, 0, clip_limit)
    
    hist_L = hist_clipped[lower_bound:Xa]
    hist_U = hist_clipped[Xa:upper_bound]
    
    N_L = np.sum(hist_L)
    N_U = np.sum(hist_U)
    
    P_L = hist_L / N_L if N_L > 0 else np.zeros_like(hist_L)
    P_U = hist_U / N_U if N_U > 0 else np.zeros_like(hist_U)
    
    C_L = np.cumsum(P_L)
    C_U = np.cumsum(P_U)
    
    # Transfer functions mapping to corresponding ranges
    # F_L maps [lower_bound, Xa) -> [lower_bound, Xa)
    F_L = np.asarray(lower_bound + (Xa - lower_bound - 1) * C_L, dtype=np.float32)
    # F_U maps [Xa, upper_bound) -> [Xa, upper_bound)
    F_U = np.asarray(Xa + (upper_bound - Xa - 1) * C_U, dtype=np.float32)
    
    img_mapped = np.zeros_like(img_gray, dtype=np.float32)
    
    lower_mask = mask & (img_gray < Xa)
    upper_mask = mask & (img_gray >= Xa)
    
    # Safely handle empty masks
    if np.any(lower_mask):
        val_idx = img_gray[lower_mask] - lower_bound
        val_idx = np.clip(val_idx, 0, len(F_L) - 1).astype(int)
        img_mapped[lower_mask] = F_L[val_idx]
        
    if np.any(upper_mask):
        val_idx = img_gray[upper_mask] - Xa
        val_idx = np.clip(val_idx, 0, len(F_U) - 1).astype(int)
        img_mapped[upper_mask] = F_U[val_idx]
        
    final_image = np.where(mask, img_mapped, img_gray)
    return np.asarray(final_image, dtype=np.uint8), np.asarray(img_mapped, dtype=np.uint8)


def apply_esihe(img_gray: np.ndarray, lower_bound=0, upper_bound=256) -> tuple[np.ndarray, float]:
    mask = (img_gray >= lower_bound) & (img_gray < upper_bound)
    if not np.any(mask):
        return img_gray.copy(), 0.0
        
    hist = cv2.calcHist([img_gray[mask]], [0], None, [256], [0, 256]).flatten()
    exposure = calculate_exposure(hist[lower_bound:upper_bound])
    
    range_span = upper_bound - lower_bound
    Xa = int(lower_bound + range_span * (1 - exposure))
    Xa = max(lower_bound + 1, min(Xa, upper_bound - 1))
    
    Tc = np.sum(hist) / range_span
    
    final_img, _ = process_sub_histogram(img_gray, lower_bound, upper_bound, Xa, Tc)
    return final_img, exposure


def apply_rs_esihe_step(img_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Applies one iteration of RS-ESIHE: computes Xa, then Xal, Xau, and equalizes 4 regions."""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    exposure = calculate_exposure(hist)
    Xa = int(L * (1 - exposure))
    Xa = max(1, min(Xa, L - 2))
    
    Tc = np.sum(hist) / L
    
    hist_lower = hist[0:Xa]
    hist_upper = hist[Xa:L]
    
    exposure_L = calculate_exposure(hist_lower)
    exposure_U = calculate_exposure(hist_upper)
    
    Xal = int(Xa * (1 - exposure_L))
    Xau = int(Xa + (L - Xa) * (1 - exposure_U))
    
    Xal = max(1, min(Xal, Xa - 1))
    Xau = max(Xa + 1, min(Xau, L - 1))

    # Apply process_sub_histogram twice: [0, Xa) split by Xal, and [Xa, L) split by Xau
    img_mapped = img_gray.copy()
    
    final_1, _ = process_sub_histogram(img_mapped, 0, Xa, Xal, Tc)
    img_mapped = np.where(img_gray < Xa, final_1, img_mapped)
    
    final_2, _ = process_sub_histogram(img_mapped, Xa, L, Xau, Tc)
    img_mapped = np.where(img_gray >= Xa, final_2, img_mapped)
    
    return np.asarray(img_mapped, dtype=np.uint8), float(exposure)


def apply_r_esihe(img_gray: np.ndarray, epsilon: float = 0.01, adaptive: bool = True) -> np.ndarray:
    max_iter = get_adaptive_max_iter(img_gray) if adaptive else 10
    current_img = img_gray.copy()
    prev_exposure = -1.0

    for _ in range(max_iter):
        enhanced_img, current_exposure = apply_esihe(current_img)
        if prev_exposure != -1.0 and abs(current_exposure - prev_exposure) < epsilon:
            break
        current_img = enhanced_img
        prev_exposure = current_exposure

    return current_img

def apply_rs_esihe(img_gray: np.ndarray, epsilon: float = 0.01, adaptive: bool = True) -> np.ndarray:
    max_iter = get_adaptive_max_iter(img_gray) if adaptive else 10
    current_img = img_gray.copy()
    prev_exposure = -1.0

    for _ in range(max_iter):
        enhanced_img, current_exposure = apply_rs_esihe_step(current_img)
        if prev_exposure != -1.0 and abs(current_exposure - prev_exposure) < epsilon:
            break
        current_img = enhanced_img
        prev_exposure = current_exposure

    return current_img

def apply_hybrid_mode(img_bgr: np.ndarray) -> np.ndarray:
    # 1. CLAHE in LAB space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # 2. Gamma Correction
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_gamma = cv2.LUT(l_clahe, table)
    
    # 3. Standard HE
    l_he = cv2.equalizeHist(l_gamma)
    
    lab_enhanced = cv2.merge([l_he, a, b])
    img_hybrid = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return img_hybrid


def get_smart_selector_algo(img_gray: np.ndarray) -> str:
    exposure = calculate_exposure(cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten())
    entropy = calculate_entropy(img_gray)
    
    # For very dark or high complexity images, RS-ESIHE performs better
    if exposure < 0.3 or entropy > 6.0:
        return "RS-ESIHE"
    return "R-ESIHE"


def process_image_pipeline(img_bgr: np.ndarray, algorithm: str = "auto", strength: float = 1.0) -> PipelineResultDto:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Pre-Metrics
    orig_hist = cv2.calcHist([v], [0], None, [256], [0, 256]).flatten()
    orig_exposure = calculate_exposure(orig_hist)
    orig_entropy = calculate_entropy(v)
    
    selected_algo = algorithm
    if algorithm == "auto":
        selected_algo = get_smart_selector_algo(v)
        
    if selected_algo == "Hybrid":
        enhanced_bgr = apply_hybrid_mode(img_bgr)
    else:
        if selected_algo == "R-ESIHE":
            v_enhanced = apply_r_esihe(v)
        elif selected_algo == "RS-ESIHE":
            v_enhanced = apply_rs_esihe(v)
        else:
            v_enhanced = apply_r_esihe(v)  # default
            
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # Blend carefully using strength slider
    blended = cv2.addWeighted(enhanced_bgr, strength, img_bgr, 1.0 - strength, 0)
    
    # Post-Metrics
    hsv_blended = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
    _, _, v_blended = cv2.split(hsv_blended)
    final_hist = cv2.calcHist([v_blended], [0], None, [256], [0, 256]).flatten()
    final_exposure = calculate_exposure(final_hist)
    final_entropy = calculate_entropy(v_blended)
    
    return PipelineResultDto(
        image_data=blended,
        metrics=TelemetryMetricsDto(
            algorithm_used=selected_algo,
            original_exposure=float(orig_exposure),
            enhanced_exposure=float(final_exposure),
            original_entropy=float(orig_entropy),
            enhanced_entropy=float(final_entropy)
        ),
        histograms=TelemetryHistogramsDto(
            original=[int(x) for x in orig_hist],
            enhanced=[int(x) for x in final_hist]
        )
    )
