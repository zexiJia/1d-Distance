<div align="center">

# 📊 CHD & CMMS

### Evaluating Generative Models via One-Dimensional Code Distributions

<a href="https://arxiv.org/abs/2603.08064" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.08064-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/datasets/ZexiJia/Visform" target="_blank">
    <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VisForm-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" height="25" />
</a>

**🎉 This repository accompanies our accepted CVPR 2026 paper**  
**Final average review score: 5.5**

**Codebook-based evaluation toolkit for generative image models using TiTok discrete tokenization.**

</div>

---

## 🌟 Overview

This repository contains the official implementation of our **CVPR 2026 accepted paper**,  
**"Evaluating Generative Models via One-Dimensional Code Distributions"**  

It provides two complementary evaluation metrics based on [TiTok](https://github.com/bytedance/1d-tokenizer) discrete codebooks:

| Metric | Full Name | Level | Description | Direction |
|--------|-----------|-------|-------------|-----------|
| **CHD** | Codebook Histogram Distance | Distribution | Measures the Hellinger distance between codebook usage histograms of real vs. generated images | Lower is better ↓ |
| **CMMS** | Codebook-based Model Metric Score | Image | Uses a trained Transformer regressor to predict per-image quality scores from codebook vector sequences | Higher is better ↑ |

### Key Features

- **🔢 Discrete Code Space**: Operates in TiTok's 1D discrete token space rather than continuous embeddings
- **📐 Distribution-Level (CHD)**: Captures global distributional differences via codebook usage histograms
- **🖼️ Image-Level (CMMS)**: Provides fine-grained per-image quality scores via a learned Transformer regressor
- **⚡ Efficient**: Lightweight TiTok encoding enables fast batch evaluation
- **🔌 Self-Contained**: TiTok tokenizer code is bundled — just download weights and run
- **🧩 Modular**: CHD and CMMS are independent modules that can be used separately or together

---

## 🗂️ Project Structure

```text
CHD/
├── README.md               # This file
├── evaluate.py             # Unified evaluation entry point
├── requirements.txt        # Dependencies
├── chd/                    # CHD metric package
│   ├── __init__.py
│   └── chd_metric.py       # CHD computation (Hellinger distance on codebook histograms)
├── cmms/                   # CMMS metric package
│   ├── __init__.py
│   └── cmms_metric.py      # CMMS computation (Transformer regressor + codebook vectors)
├── modeling/               # TiTok tokenizer (bundled from 1d-tokenizer)
│   ├── __init__.py
│   ├── titok.py            # TiTok model definition
│   ├── modules/            # Encoder, decoder, building blocks
│   │   ├── base_model.py
│   │   ├── blocks.py
│   │   └── maskgit_vqgan.py
│   └── quantizer/          # Vector quantizer
│       └── quantizer.py
├── configs/                # TiTok model configs
│   ├── titok_l32.yaml      # TiTok-L-32 config (used by CHD)
│   └── titok_s128.yaml     # TiTok-S-128 config (used by CMMS)
├── checkpoints/            # Pre-trained model weights (download below)
└── dataset/
    └── README.md           # VisForm benchmark dataset description

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ZexiJia/CHD.git
cd CHD
pip install -r requirements.txt
```

### Download Pre-trained Weights

TiTok tokenizer weights are automatically downloaded from HuggingFace when you first run the evaluation. No manual setup is needed.

Alternatively, you can pre-download them for offline use:

```python
# CHD uses TiTok-L-32 (32 latent tokens)
from modeling.titok import TiTok
titok_l32 = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")

# CMMS uses TiTok-S-128 (128 latent tokens)
titok_s128 = TiTok.from_pretrained("yucornetto/tokenizer_titok_s128_imagenet")
```

The CMMS regressor checkpoint should be placed at:

```
checkpoints/cmms_best.pt
```

> **TiTok weights**: [yucornetto/tokenizer_titok_l32_imagenet](https://huggingface.co/yucornetto/tokenizer_titok_l32_imagenet) and [yucornetto/tokenizer_titok_s128_imagenet](https://huggingface.co/yucornetto/tokenizer_titok_s128_imagenet) on HuggingFace Hub.

---

### Compute CHD

CHD encodes images into discrete tokens via TiTok, computes codebook usage frequency histograms, and measures the Hellinger distance between distributions.

```bash
# CLI
python -m chd.chd_metric \
    --real_dir /path/to/real_images \
    --gen_dir /path/to/generated_images \
    --batch_size 128
```

```python
# Python API
from chd import compute_chd_from_folders

chd_score = compute_chd_from_folders(
    real_folder="path/to/real",
    gen_folder="path/to/generated",
    model_name_or_path="yucornetto/tokenizer_titok_l32_imagenet",
    device="cuda",
    batch_size=128,
)
print(f"CHD: {chd_score:.6f}")  # Lower is better
```

### Compute CMMS

CMMS encodes images via TiTok, looks up codebook vectors, and uses a trained Transformer regressor to predict quality scores.

```bash
# CLI
python -m cmms.cmms_metric \
    --image_dir /path/to/images \
    --ckpt checkpoints/cmms_best.pt \
    --batch_size 256
```

```python
# Python API
from cmms import compute_cmms_scores
import numpy as np

scores = compute_cmms_scores(
    image_paths=["img1.png", "img2.png"],
    cmms_ckpt_path="checkpoints/cmms_best.pt",
    device="cuda",
    batch_size=256,
)
print(f"CMMS Mean: {np.mean(scores):.6f}")  # Higher is better
```

### Unified Evaluation

```bash
# Compute both CHD and CMMS
python evaluate.py \
    --real_dir /path/to/real_images \
    --gen_dir /path/to/generated_images \
    --metrics chd cmms \
    --cmms_ckpt checkpoints/cmms_best.pt

# CHD only
python evaluate.py \
    --real_dir /path/to/real_images \
    --gen_dir /path/to/generated_images \
    --metrics chd

# CMMS only
python evaluate.py \
    --gen_dir /path/to/generated_images \
    --metrics cmms \
    --cmms_ckpt checkpoints/cmms_best.pt
```

---

## ⚙️ Parameters

### CHD Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--real_dir` | Required | Path to real image folder |
| `--gen_dir` | Required | Path to generated image folder |
| `--model` | `yucornetto/tokenizer_titok_l32_imagenet` | TiTok model name or local path |
| `--batch_size` | 128 | Batch size |
| `--codebook_size` | 4096 | TiTok codebook size |

### CMMS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image_dir` | Required | Path to image folder |
| `--ckpt` | Required | Path to CMMS regressor checkpoint |
| `--titok_model` | `yucornetto/tokenizer_titok_s128_imagenet` | TiTok model name or local path |
| `--batch_size` | 256 | Batch size |

---

## 📦 Pre-trained Weights

| Weight | Location | Source |
|--------|----------|--------|
| TiTok-L-32 tokenizer | Auto-downloaded | [yucornetto/tokenizer_titok_l32_imagenet](https://huggingface.co/yucornetto/tokenizer_titok_l32_imagenet) |
| TiTok-S-128 tokenizer | Auto-downloaded | [yucornetto/tokenizer_titok_s128_imagenet](https://huggingface.co/yucornetto/tokenizer_titok_s128_imagenet) |
| CMMS regressor | `checkpoints/cmms_best.pt` | Provided with this repo |

---

## 📊 VisForm Dataset

The **VisForm** benchmark dataset used in our paper is publicly available at:

🤗 [https://huggingface.co/datasets/ZexiJia/Visform](https://huggingface.co/datasets/ZexiJia/Visform)

See [`dataset/README.md`](dataset/README.md) for details.

---

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NumPy, Pillow, tqdm, einops, omegaconf
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (for auto-downloading TiTok weights)

---

## 📝 Citation

```bibtex
@article{jia2026evaluating,
  title={Evaluating Generative Models via One-Dimensional Code Distributions},
  author={Jia, Zexi and Luo, Pengcheng and Zhong, Yijia and Zhang, Jinchao and Zhou, Jie},
  journal={arXiv preprint arXiv:2603.08064},
  year={2026}
}
```

---

## 🙏 Acknowledgments

- **[TiTok](https://github.com/bytedance/1d-tokenizer)**: 1D discrete image tokenizer backbone (Apache 2.0 License, Copyright 2024 Bytedance Ltd.)
- **[VisForm](https://huggingface.co/datasets/ZexiJia/Visform)**: Large-scale benchmark for evaluating generative image models

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

The bundled `modeling/` directory is adapted from the [1d-tokenizer](https://github.com/bytedance/1d-tokenizer) project, also under the Apache License 2.0.
