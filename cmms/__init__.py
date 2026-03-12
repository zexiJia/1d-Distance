"""
CMMS (Codebook-based Model Metric Score) Package

An image-level evaluation metric for generative image models.
CMMS encodes images into discrete tokens via TiTok, looks up codebook vectors,
and uses a trained Transformer regressor to predict per-image quality scores.
"""

from cmms.cmms_metric import (
    compute_cmms_scores,
    ScoreRegressor,
)

__all__ = [
    "compute_cmms_scores",
    "ScoreRegressor",
]
