import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_image(filepath: str) -> np.ndarray:
    """
    Loads an image from a path.

    Args:
        filepath (str): Image path.

    Returns:
        np.ndarray: Loaded BGR image.

    Raises:
        FileNotFoundError: If the image cannot be found/loaded.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {filepath}")
    return img


def save_image(filepath: str, img: np.ndarray):
    """
    Saves an image to a path, creating parent directories if they do not exist.

    Args:
        filepath (str): Target save path.
        img (np.ndarray): BGR image array.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    cv2.imwrite(filepath, img)


def plot_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    target: np.ndarray | None = None,
    title: str = "Comparison",
    save_path: str | None = None
):
    """
    Plots a side-by-side comparison of the images using matplotlib.

    Args:
        original (np.ndarray): BGR Original image.
        enhanced (np.ndarray): BGR Enhanced image.
        target (np.ndarray, optional): BGR Ground truth image.
        title (str): Figure title.
        save_path (str, optional): Where to save the figure if specified.
    """
    num_cols = 3 if target is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # Convert BGR to RGB for matplotlib
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original (Low Light)")
    axes[0].axis('off')

    axes[1].imshow(enh_rgb)
    axes[1].set_title("Enhanced (R-ESIHE)")
    axes[1].axis('off')

    if target is not None:
        tgt_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        axes[2].imshow(tgt_rgb)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)
