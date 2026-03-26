import argparse
import os
import sys
from src.core.enhancer import process_color_image
from src.utils.metrics import calculate_metrics
from src.utils.image_io import load_image, save_image, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhancement of Low Exposure Images via Recursive Histogram (R-ESIHE)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input low-light image or a directory of images.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to save the enhanced output image or directory.")
    parser.add_argument("--target", "-t", type=str, default=None,
                        help="Optional. Path to ground truth image/directory for PSNR/SSIM calculation.")
    parser.add_argument("--epsilon", "-e", type=float, default=0.01,
                        help="Convergence threshold for exposure (default: 0.01)")
    parser.add_argument("--max_iter", "-m", type=int, default=10,
                        help="Maximum number of recursion iterations (default: 10)")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Enable to show/save a side-by-side comparison plot.")
    return parser.parse_args()


def process_single_image(input_path: str, output_path: str, target_path: str | None = None,
                         epsilon: float = 0.01, max_iter: int = 10, plot: bool = False):

    print(f"Processing: {input_path}")
    try:
        img_bgr = load_image(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Enhance the image
    enhanced_bgr, iters = process_color_image(
        img_bgr, epsilon=epsilon, max_iter=max_iter)

    # Save output
    save_image(output_path, enhanced_bgr)
    print(
        f"  -> Saved output to {output_path} (Converged in {iters} iterations)")

    # Compute metrics if target provided
    target_bgr = None
    if target_path and os.path.exists(target_path):
        target_bgr = load_image(target_path)
        psnr, ssim_val = calculate_metrics(target_bgr, enhanced_bgr)
        print(f"  -> Metrics - PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")

    if plot:
        plot_name = os.path.splitext(output_path)[0] + "_plot.png"
        plot_comparison(img_bgr, enhanced_bgr, target_bgr,
                        title=f"Result (Iters: {iters})", save_path=plot_name)
        print(f"  -> Saved comparison plot to {plot_name}")


def main():
    args = parse_args()

    # Check if input is a directory
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(
            args.input) if f.lower().endswith(valid_exts)]

        if not image_files:
            print(f"No valid images found in directory {args.input}")
            sys.exit(0)

        for img_file in image_files:
            in_path = os.path.join(args.input, img_file)
            out_path = os.path.join(args.output, img_file)

            tgt_path = None
            if args.target and os.path.isdir(args.target):
                tgt_path = os.path.join(args.target, img_file)

            process_single_image(in_path, out_path, tgt_path,
                                 epsilon=args.epsilon, max_iter=args.max_iter, plot=args.plot)
    else:
        # Single image logic
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)

        process_single_image(args.input, args.output, args.target,
                             epsilon=args.epsilon, max_iter=args.max_iter, plot=args.plot)


if __name__ == "__main__":
    main()
