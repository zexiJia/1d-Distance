#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CHD (Codebook Histogram Distance) Module

Encodes images into discrete token sequences via TiTok, computes codebook
usage frequency histograms, and measures the Hellinger distance between
real and generated image distributions.

Lower CHD values indicate that the generated distribution is closer to the real one.
"""

import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ========= Utility Functions =========

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    """List all image files in a folder."""
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return sorted(paths)


def load_image_tensor(path, size=(256, 256)):
    """Load an image and convert to tensor [1, 3, H, W]."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


# ========= TiTok Model Loading =========

def load_titok(model_name_or_path="yucornetto/tokenizer_titok_l32_imagenet", device="cuda"):
    """
    Load the TiTok tokenizer model.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        device: Compute device.

    Returns:
        Loaded TiTok model.
    """
    from modeling.titok import TiTok
    tokenizer = TiTok.from_pretrained(model_name_or_path)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)
    return tokenizer


# ========= Codebook Histogram Computation =========

@torch.no_grad()
def accumulate_code_histogram(
    titok_tokenizer,
    image_paths,
    device="cuda",
    batch_size=128,
    resize=(256, 256),
    codebook_size=4096
):
    """
    Compute the TiTok codebook usage histogram for a set of images.

    Args:
        titok_tokenizer: TiTok tokenizer model.
        image_paths: List of image file paths.
        device: Compute device.
        batch_size: Batch size for encoding.
        resize: Image resize dimensions.
        codebook_size: Size of the TiTok codebook.

    Returns:
        (p, N): Normalized histogram distribution, codebook size.
    """
    counts = torch.zeros(codebook_size, dtype=torch.float64, device=device)

    for s in tqdm(range(0, len(image_paths), batch_size), desc="Computing code histogram"):
        batch_paths = image_paths[s:s + batch_size]
        imgs = [load_image_tensor(p, size=resize) for p in batch_paths]
        batch = torch.cat(imgs, dim=0).to(device)
        encoded_tokens = titok_tokenizer.encode(batch)[1]["min_encoding_indices"]
        idx = encoded_tokens.squeeze(1) if encoded_tokens.dim() > 2 else encoded_tokens
        counts += torch.bincount(idx.reshape(-1), minlength=codebook_size).to(torch.float64)

    p = (counts / counts.sum()).detach().cpu().numpy()
    return p, codebook_size


# ========= Core CHD Computation =========

def compute_chd(p_real, p_gen):
    """
    Compute the CHD (Codebook Histogram Distance) metric.

    Based on the Hellinger distance between two codebook histogram distributions.
    Range: [0, 1], where 0 means identical distributions and 1 means completely different.

    Args:
        p_real: Normalized codebook histogram of real images (probability vector).
        p_gen: Normalized codebook histogram of generated images (probability vector).

    Returns:
        float: CHD metric value.
    """
    p_r = np.array(p_real, dtype=np.float64).reshape(-1)
    p_g = np.array(p_gen, dtype=np.float64).reshape(-1)

    # Ensure distributions are non-negative and normalized
    p_r = np.maximum(p_r, 0)
    p_g = np.maximum(p_g, 0)
    p_r = p_r / p_r.sum() if p_r.sum() > 0 else p_r
    p_g = p_g / p_g.sum() if p_g.sum() > 0 else p_g

    # Compute Hellinger distance
    hellinger_distance = np.sqrt(np.sum((np.sqrt(p_r) - np.sqrt(p_g)) ** 2)) / np.sqrt(2)

    return float(hellinger_distance)


def compute_chd_from_paths(
    real_image_paths,
    gen_image_paths,
    titok_tokenizer=None,
    model_name_or_path="yucornetto/tokenizer_titok_l32_imagenet",
    device="cuda",
    batch_size=128,
    resize=(256, 256),
    codebook_size=4096,
):
    """
    Compute CHD directly from lists of image paths.

    Args:
        real_image_paths: List of real image paths.
        gen_image_paths: List of generated image paths.
        titok_tokenizer: TiTok tokenizer model (optional; auto-loaded if None).
        model_name_or_path: TiTok model name or path (used only if titok_tokenizer is None).
        device: Compute device.
        batch_size: Batch size for encoding.
        resize: Image resize dimensions.
        codebook_size: Size of the TiTok codebook.

    Returns:
        float: CHD metric value.
    """
    if titok_tokenizer is None:
        titok_tokenizer = load_titok(model_name_or_path, device)

    print("Computing real images histogram...")
    p_real, N_real = accumulate_code_histogram(
        titok_tokenizer, real_image_paths,
        device=device, batch_size=batch_size, resize=resize, codebook_size=codebook_size
    )

    print("Computing generated images histogram...")
    p_gen, N_gen = accumulate_code_histogram(
        titok_tokenizer, gen_image_paths,
        device=device, batch_size=batch_size, resize=resize, codebook_size=codebook_size
    )

    assert N_real == N_gen, f"Codebook size mismatch: {N_real} vs {N_gen}"

    print("Computing CHD score...")
    chd_score = compute_chd(p_real, p_gen)
    return chd_score


def compute_chd_from_folders(
    real_folder,
    gen_folder,
    titok_tokenizer=None,
    model_name_or_path="yucornetto/tokenizer_titok_l32_imagenet",
    device="cuda",
    batch_size=128,
    resize=(256, 256),
    codebook_size=4096,
):
    """
    Compute CHD from folder paths.

    Args:
        real_folder: Path to the folder of real images.
        gen_folder: Path to the folder of generated images.
        titok_tokenizer: TiTok tokenizer model (optional).
        model_name_or_path: TiTok model name or path.
        device: Compute device.
        batch_size: Batch size for encoding.
        resize: Image resize dimensions.
        codebook_size: Size of the TiTok codebook.

    Returns:
        float: CHD metric value.
    """
    real_paths = list_images(real_folder)
    gen_paths = list_images(gen_folder)

    assert len(real_paths) > 0, f"No images found in real folder: {real_folder}"
    assert len(gen_paths) > 0, f"No images found in gen folder: {gen_folder}"

    print(f"Found {len(real_paths)} real images and {len(gen_paths)} generated images")

    return compute_chd_from_paths(
        real_paths, gen_paths, titok_tokenizer, model_name_or_path,
        device, batch_size, resize, codebook_size
    )


# ========= CLI Entry Point =========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute CHD (Codebook Histogram Distance)")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to real image folder")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated image folder")
    parser.add_argument("--model", type=str, default="yucornetto/tokenizer_titok_l32_imagenet",
                        help="TiTok model name or local path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Image resize dimensions")
    parser.add_argument("--codebook_size", type=int, default=4096, help="Codebook size")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")

    args = parser.parse_args()

    chd = compute_chd_from_folders(
        real_folder=args.real_dir,
        gen_folder=args.gen_dir,
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        resize=tuple(args.resize),
        codebook_size=args.codebook_size,
    )
    print(f"\nCHD Score: {chd:.6f}")
