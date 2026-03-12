#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py - Unified Evaluation Script

Supports computing both CHD and CMMS metrics.

Usage:
    # Compute CHD only
    python evaluate.py --real_dir /path/to/real --gen_dir /path/to/gen --metrics chd

    # Compute CMMS only
    python evaluate.py --gen_dir /path/to/gen --metrics cmms --cmms_ckpt best.pt

    # Compute both metrics
    python evaluate.py --real_dir /path/to/real --gen_dir /path/to/gen --metrics chd cmms --cmms_ckpt best.pt
"""

import argparse
import os
import glob
import numpy as np


def collect_images(folder, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")):
    """Collect all image paths from a folder."""
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def run_chd(args):
    """Compute the CHD metric."""
    from chd.chd_metric import compute_chd_from_folders

    print("\n" + "=" * 60)
    print("  Computing CHD (Codebook Histogram Distance)")
    print("=" * 60)

    chd_score = compute_chd_from_folders(
        real_folder=args.real_dir,
        gen_folder=args.gen_dir,
        model_name_or_path=args.titok_model_chd,
        device=args.device,
        batch_size=args.batch_size,
        resize=tuple(args.resize),
        codebook_size=args.codebook_size,
    )
    print(f"\n  CHD Score: {chd_score:.6f}")
    return {"CHD": chd_score}


def run_cmms(args):
    """Compute the CMMS metric."""
    from cmms.cmms_metric import compute_cmms_scores

    print("\n" + "=" * 60)
    print("  Computing CMMS (Codebook-based Model Metric Score)")
    print("=" * 60)

    image_paths = collect_images(args.gen_dir)
    assert len(image_paths) > 0, f"No images found in: {args.gen_dir}"
    print(f"  Found {len(image_paths)} images")

    scores = compute_cmms_scores(
        image_paths=image_paths,
        cmms_ckpt_path=args.cmms_ckpt,
        titok_model_name=args.titok_model_cmms,
        device=args.device,
        batch_size=args.batch_size,
        resize=tuple(args.resize),
    )

    valid_scores = [s for s in scores if not np.isnan(s)]
    mean_score = np.mean(valid_scores) if valid_scores else 0.0
    std_score = np.std(valid_scores) if valid_scores else 0.0

    print(f"\n  CMMS Mean: {mean_score:.6f} ± {std_score:.6f}")
    print(f"  Valid images: {len(valid_scores)}/{len(scores)}")
    return {"CMMS_mean": mean_score, "CMMS_std": std_score, "CMMS_valid": len(valid_scores)}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Script - Compute CHD / CMMS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common arguments
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Path to real image folder (required for CHD)")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Path to generated image folder")
    parser.add_argument("--metrics", type=str, nargs="+", default=["chd"],
                        choices=["chd", "cmms"],
                        help="Metrics to compute (can specify multiple)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Image resize dimensions")

    # CHD-specific arguments
    parser.add_argument("--titok_model_chd", type=str,
                        default="yucornetto/tokenizer_titok_l32_imagenet",
                        help="TiTok model name or path for CHD")
    parser.add_argument("--codebook_size", type=int, default=4096, help="Codebook size")

    # CMMS-specific arguments
    parser.add_argument("--cmms_ckpt", type=str, default=None,
                        help="Path to CMMS model checkpoint")
    parser.add_argument("--titok_model_cmms", type=str,
                        default="yucornetto/tokenizer_titok_s128_imagenet",
                        help="TiTok model name or path for CMMS")

    args = parser.parse_args()

    # Argument validation
    if "chd" in args.metrics and args.real_dir is None:
        parser.error("CHD requires --real_dir argument")
    if "cmms" in args.metrics and args.cmms_ckpt is None:
        parser.error("CMMS requires --cmms_ckpt argument")

    results = {}

    if "chd" in args.metrics:
        results.update(run_chd(args))
    if "cmms" in args.metrics:
        results.update(run_cmms(args))

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
