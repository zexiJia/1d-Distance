"""
CHD (Codebook Histogram Distance) Package

A distribution-level evaluation metric for generative image models.
CHD encodes images into discrete tokens via TiTok, computes codebook usage
histograms, and measures the Hellinger distance between real and generated
distributions.
"""

from chd.chd_metric import (
    compute_chd,
    compute_chd_from_paths,
    compute_chd_from_folders,
)

__all__ = [
    "compute_chd",
    "compute_chd_from_paths",
    "compute_chd_from_folders",
]
