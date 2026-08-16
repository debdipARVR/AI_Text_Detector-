"""Engine package for Cloze Congruence AI Text Detection and Humanization."""

from .cloze_masker import ClozeMasker, ClozeMaskResult, MaskedSpan
from .detector import ClozeCongruenceDetector
from .humanizer import TextHumanizer, HUMANIZER_MODES
from .metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_ai_probability,
    compute_lexical_similarity,
    compute_semantic_congruence,
)
from .nim_client import NvidiaNIMClient, NVIDIA_MODELS

__all__ = [
    "ClozeMasker",
    "ClozeMaskResult",
    "MaskedSpan",
    "ClozeCongruenceDetector",
    "TextHumanizer",
    "HUMANIZER_MODES",
    "NvidiaNIMClient",
    "NVIDIA_MODELS",
    "compute_lexical_similarity",
    "compute_semantic_congruence",
    "calculate_burstiness",
    "compute_ai_probability",
    "classify_span_congruence",
]
