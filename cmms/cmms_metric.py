#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMMS (Codebook-based Model Metric Score) Module

Encodes images into discrete token sequences via TiTok, looks up codebook
vectors to build embedding sequences, and uses a trained Transformer regressor
to predict per-image quality scores.

Pipeline: Image -> TiTok Encode -> Token Indices -> Codebook Lookup -> Transformer -> Score
"""

import os
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn


# ========= Utility Functions =========

def load_image_tensor(path, size=(256, 256)):
    """Load an image and convert to tensor [1, 3, H, W]."""
    try:
        img = Image.open(path).convert("RGB").resize(size)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    except Exception:
        return None


# ========= TiTok Model Utilities =========

def load_titok(device, model_name_or_path="yucornetto/tokenizer_titok_s128_imagenet"):
    """
    Load the TiTok tokenizer.

    Args:
        device: Compute device.
        model_name_or_path: HuggingFace model name or local path.

    Returns:
        Loaded TiTok model.
    """
    from modeling.titok import TiTok
    tok = TiTok.from_pretrained(model_name_or_path)
    tok.eval().requires_grad_(False)
    return tok.to(device)


@torch.no_grad()
def titok_encode_indices(titok, img_batch):
    """
    Encode an image batch with TiTok and return token indices.

    Args:
        titok: TiTok model.
        img_batch: [B, 3, H, W] image tensor.

    Returns:
        [B, N] token index tensor.
    """
    out = titok.encode(img_batch)
    if isinstance(out, (tuple, list)):
        d = next((x for x in out if isinstance(x, dict)), None)
        d = d if d is not None else (out[0] if len(out) > 0 and isinstance(out[0], dict) else {})
    elif isinstance(out, dict):
        d = out
    else:
        d = getattr(out, "__dict__", {})

    for k in ["min_encoding_indices", "encoding_indices", "indices", "codes"]:
        if k in d:
            idx = d[k]
            if isinstance(idx, torch.Tensor):
                return idx if idx.dim() == 2 else idx.view(idx.size(0), -1)
            if isinstance(idx, np.ndarray):
                idx = torch.from_numpy(idx)
                return idx if idx.dim() == 2 else idx.reshape(idx.shape[0], -1)

    for v in d.values():
        if isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64):
            return v if v.dim() == 2 else v.view(v.size(0), -1)

    raise RuntimeError("Failed to find token indices in TiTok encode output.")


def titok_codebook_matrix(titok):
    """
    Extract the codebook weight matrix from TiTok.

    Args:
        titok: TiTok model.

    Returns:
        [V, D] codebook matrix.
    """
    cand = ["codebook", "embedding", "embeddings", "embed", "quantizer",
            "quantize", "vq", "vector_quantizer"]

    if hasattr(titok, "get_codebook"):
        try:
            W = titok.get_codebook()
            if isinstance(W, torch.Tensor):
                return W.detach().float().cpu()
        except Exception:
            pass

    def _maybe_weight(mod):
        for name in ["weight", "codebook", "embedding_weight"]:
            if hasattr(mod, name):
                W = getattr(mod, name)
                if isinstance(W, torch.Tensor) and W.dim() == 2:
                    return W
        return None

    w = _maybe_weight(titok)
    if w is not None:
        return w.detach().float().cpu()

    for n, m in titok.named_modules():
        if any(a in n.lower() for a in cand):
            w = _maybe_weight(m)
            if w is not None:
                return w.detach().float().cpu()
    for _, m in titok.named_modules():
        w = _maybe_weight(m)
        if w is not None:
            return w.detach().float().cpu()

    raise RuntimeError("Failed to find TiTok codebook matrix.")


# ========= Transformer Regressor Model =========

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionPool(nn.Module):
    """Attention pooling layer."""
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, D = x.size()
        q = self.q.expand(B, -1, -1)
        k = v = self.proj(x)
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(D)
        attn = attn.masked_fill(~mask.unsqueeze(1), -1e4)
        w = torch.softmax(attn, dim=-1)
        pooled = torch.matmul(w, v).squeeze(1)
        return pooled


class ScoreRegressor(nn.Module):
    """
    Transformer-based image quality score regressor.

    Takes codebook vector sequences as input and outputs quality scores.

    Args:
        d_in: Input dimension (codebook vector dimension).
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_ff: Feed-forward network dimension.
        dropout: Dropout rate.
    """
    def __init__(self, d_in, nhead=4, num_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d_in)
        self.pos = PositionalEncoding(d_in)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_in, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPool(d_in)
        self.head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, 1)
        )

    def forward(self, x, key_padding_mask):
        x = self.in_proj(self.in_norm(x))
        x = self.pos(x)
        enc = self.encoder(x, src_key_padding_mask=~key_padding_mask)
        pooled = self.pool(enc, key_padding_mask)
        logits = self.head(pooled).squeeze(-1)
        return logits


def load_regressor(ckpt_path, d_model, device, nhead=4, num_layers=4, dim_ff=512, dropout=0.1):
    """
    Load a trained ScoreRegressor model.

    Args:
        ckpt_path: Path to checkpoint file.
        d_model: Model dimension.
        device: Compute device.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_ff: Feed-forward network dimension.
        dropout: Dropout rate.

    Returns:
        Loaded model in eval mode.
    """
    model = ScoreRegressor(
        d_in=d_model, nhead=nhead, num_layers=num_layers,
        dim_ff=dim_ff, dropout=dropout
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ========= Sequence Padding Utilities =========

def pad_and_make_mask(seqs):
    """
    Pad variable-length sequences to the same length and generate masks.

    Args:
        seqs: List of tensors, each [T_i, D].

    Returns:
        (x, m): x is [B, T, D] padded tensor, m is [B, T] boolean mask.
    """
    lengths = [s.size(0) for s in seqs]
    D = seqs[0].size(1)
    T = max(lengths)
    B = len(seqs)
    x = torch.zeros(B, T, D, dtype=torch.float32)
    m = torch.zeros(B, T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        t = s.size(0)
        x[i, :t] = s
        m[i, :t] = True
    return x, m


# ========= Core CMMS Computation =========

@torch.no_grad()
def compute_cmms_scores(
    image_paths,
    titok=None,
    codebook=None,
    model=None,
    titok_model_name="yucornetto/tokenizer_titok_s128_imagenet",
    cmms_ckpt_path=None,
    device="cuda",
    batch_size=256,
    resize=(256, 256),
    max_tokens=None,
    nhead=4,
    num_layers=4,
    dim_ff=512,
    dropout=0.1,
):
    """
    Compute CMMS quality scores for a set of images.

    Args:
        image_paths: List of image file paths.
        titok: TiTok model (optional; auto-loaded if None).
        codebook: Codebook matrix [V, D] (optional; auto-extracted if None).
        model: ScoreRegressor model (optional; requires cmms_ckpt_path if None).
        titok_model_name: TiTok model name (used only if titok is None).
        cmms_ckpt_path: Path to CMMS model checkpoint.
        device: Compute device.
        batch_size: Batch size for encoding.
        resize: Image resize dimensions.
        max_tokens: Maximum token count limit.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_ff: Feed-forward network dimension.
        dropout: Dropout rate.

    Returns:
        List of scores, one float per image (NaN for failed images).
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Auto-load models
    if titok is None:
        titok = load_titok(device, titok_model_name)
    if codebook is None:
        codebook = titok_codebook_matrix(titok)
    V, D = codebook.shape
    if model is None:
        assert cmms_ckpt_path is not None, "Must provide cmms_ckpt_path or a pre-loaded model"
        model = load_regressor(cmms_ckpt_path, d_model=D, device=device,
                               nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, dropout=dropout)

    scores = []
    for s in tqdm(range(0, len(image_paths), batch_size), desc="CMMS scoring"):
        batch_paths = image_paths[s:s + batch_size]
        img_list, good_idx = [], []
        for i, p in enumerate(batch_paths):
            t = load_image_tensor(p, size=resize)
            if t is not None:
                img_list.append(t)
                good_idx.append(i)
        if not img_list:
            scores.extend([np.nan] * len(batch_paths))
            continue

        imgs = torch.cat(img_list, dim=0).to(device, non_blocking=True)
        idxs = titok_encode_indices(titok, imgs)

        seqs = []
        for j in range(idxs.size(0)):
            idx = idxs[j].detach().long().cpu().clamp_(0, V - 1)
            if max_tokens is not None and idx.numel() > max_tokens:
                idx = idx[:max_tokens]
            vec = codebook.index_select(0, idx).float()
            seqs.append(vec)

        x_cpu, m_cpu = pad_and_make_mask(seqs)
        x = x_cpu.to(device, non_blocking=True)
        m = m_cpu.to(device, non_blocking=True)

        logits = model(x, m)
        y = torch.sigmoid(logits).detach().float().cpu().numpy().tolist()

        out_batch = [np.nan] * len(batch_paths)
        for k, gi in enumerate(good_idx):
            out_batch[gi] = float(y[k])
        scores.extend(out_batch)

        del imgs, idxs, x, m, x_cpu, m_cpu, seqs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return scores


# ========= CLI Entry Point =========

if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="Compute CMMS (Codebook-based Model Metric Score)")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to CMMS model checkpoint")
    parser.add_argument("--titok_model", type=str, default="yucornetto/tokenizer_titok_s128_imagenet",
                        help="TiTok model name or local path")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Image resize dimensions")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")

    args = parser.parse_args()

    # Collect images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_paths = sorted(image_paths)

    assert len(image_paths) > 0, f"No images found in: {args.image_dir}"
    print(f"Found {len(image_paths)} images")

    scores = compute_cmms_scores(
        image_paths=image_paths,
        cmms_ckpt_path=args.ckpt,
        titok_model_name=args.titok_model,
        device=args.device,
        batch_size=args.batch_size,
        resize=tuple(args.resize),
    )

    valid_scores = [s for s in scores if not np.isnan(s)]
    print(f"\nCMMS Results:")
    print(f"  Valid images: {len(valid_scores)}/{len(scores)}")
    print(f"  Mean score:   {np.mean(valid_scores):.6f}")
    print(f"  Std score:    {np.std(valid_scores):.6f}")
