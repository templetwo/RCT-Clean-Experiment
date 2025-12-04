"""
RCT: Relational Coherence Training

Training a language model using relational coherence rather than RLHF.

Authors: Anthony J. Vasquez Sr. & Claude
Date: December 2025
"""

from .relational_loss import (
    RelationalCoherenceLoss,
    RelationalCoherenceTracker,
    compute_relational_coherence
)
from .dataset import (
    load_relational_corpus,
    RelationalDataCollator,
    create_sample_corpus
)

__version__ = "0.1.0"
__all__ = [
    "RelationalCoherenceLoss",
    "RelationalCoherenceTracker", 
    "compute_relational_coherence",
    "load_relational_corpus",
    "RelationalDataCollator",
    "create_sample_corpus"
]
