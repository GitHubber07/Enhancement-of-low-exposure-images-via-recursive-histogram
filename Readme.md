# Enhancement of Low Exposure Images via Recursive Histogram

## Overview

This project implements **Recursive Exposure-based Sub-Image Histogram Equalization (R-ESIHE)** for enhancing low-light images. It extends the standard ESIHE algorithm by applying it recursively to achieve improved exposure correction while preserving image details and naturally balancing the illumination.

Traditional histogram equalization techniques often lead to over-enhancement, color distortion, or loss of natural appearance. This method ensures controlled enhancement through an iterative, value-channel focused processing approach.

## Key Features

- **Exposure-based Sub-Image Histogram Equalization (ESIHE):** Partitioning histograms to limit over-equalization.
- **Recursive Enhancement (R-ESIHE):** Automatically runs iteratively until exposure correction converges.
- **Evaluation Metrics:** Includes functions to automatically calculate **PSNR** & **SSIM** when a ground-truth is available.
- **Modular Python Structure:** Clean, maintainable code separated into core algorithms, utilities, and a robust Command Line Interface (CLI).

## Project Structure

```bash
.
├── data/
│   ├── input/              # Place your low-light images here
│   └── output/             # Enhanced images will be saved here
├── src/
│   ├── core/
│   │   └── enhancer.py     # Core R-ESIHE and evaluation algorithms
│   └── utils/
│   │   ├── image_io.py     # Image loading and comparison plotting
│       └── metrics.py      # PSNR and SSIM calculations
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── Recursive_ESIHE.ipynb   # Original exploratory notebook
└── Readme.md               # Project documentation
```

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage (CLI)

You can run the enhancement tool on a single image or an entire directory of images.

### 1. Enhance a Single Image
```bash
python main.py --input data/input/sample.png --output data/output/sample_enhanced.png
```

### 2. Enhance an Entire Directory
```bash
python main.py --input data/input --output data/output
```

### 3. Compare with Ground Truth (Calculate PSNR / SSIM)
If you have high-quality versions of the images, you can evaluate the model structure automatically.
```bash
python main.py --input data/input/sample.png --output data/output/sample.png --target data/gt/sample.png
```

### 4. Visualize the Enhancement Side-by-Side
Use the `--plot` flag to automatically save a matplotlib comparison grid containing the Original, the Enhanced R-ESIHE image, and the Ground Truth (if provided).
```bash
python main.py --input data/input --output data/output --plot
```

## CLI Arguments

| Parameter    | Flag | Description | Default |
|--------------|------|-------------|---------|
| `--input`    | `-i` | Path to the input low-light image or directory. | **Required** |
| `--output`   | `-o` | Path to save the enhanced output image or directory. | **Required** |
| `--target`   | `-t` | Optional. Path to ground truth image/dir for PSNR/SSIM calculation. | `None` |
| `--epsilon`  | `-e` | Convergence threshold for recursion. | `0.01` |
| `--max_iter` | `-m` | Maximum number of iterations to prevent infinite loops. | `10` |
| `--plot`     | `-p` | Flag to generate and save a side-by-side visual comparison plot. | `False` |

## Observations & Implementation Details
- **Value Channel Focus:** To preserve correct colors and saturation, R-ESIHE strictly applies computations to the **V channel** of the **HSV** color space.
- **Recursive Processing:** Progressive enhancement improves visibility smoothly.
- **Stopping Criterion:** Recursion stops when the difference between successive exposure levels is smaller than `epsilon`, effectively halting unnatural over-enhancement.

## Future Work
- Integration with deep learning-based methods (e.g., Zero-DCE, LLNet) for hybrid approaches.
- Real-time low-light video enhancement.

## Author
This algorithmic pipeline was developed as a comprehensive exploration in classical image enhancement techniques using localized histogram manipulation.