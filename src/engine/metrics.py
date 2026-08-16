"""Linguistic Congruence and Similarity Metrics for AI Detection.

Computes multi-dimensional congruence between original masked text and LLM infillings:
1. Lexical Similarity (Jaccard token overlap, RapidFuzz Levenshtein, ROUGE-L LCS)
2. Semantic Congruence (Contextual semantic equivalence and phrasing alignment)
3. Stylometric Burstiness (Sentence length variance and structural entropy)
4. Calibrated AI Probability Percentage (%)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple
from rapidfuzz.distance import Levenshtein


def tokenize_words(text: str) -> List[str]:
    """Extract lowercase word tokens from text."""
    return re.findall(r'\b[a-zA-Z0-9_\'-]+\b', text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate token-level Jaccard similarity coefficient (0.0 to 1.0)."""
    tokens1 = set(tokenize_words(text1))
    tokens2 = set(tokenize_words(text2))
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


def levenshtein_similarity(text1: str, text2: str) -> float:
    """Calculate normalized Levenshtein edit distance similarity (0.0 to 1.0)."""
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return Levenshtein.normalized_similarity(t1, t2)


def longest_common_subsequence_ratio(text1: str, text2: str) -> float:
    """Calculate ROUGE-L like Longest Common Subsequence ratio (0.0 to 1.0)."""
    tokens1 = tokenize_words(text1)
    tokens2 = tokenize_words(text2)
    m, n = len(tokens1), len(tokens2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if tokens1[i] == tokens2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / n
    rec = lcs_len / m
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """Composite lexical similarity score combining Jaccard, Levenshtein, and LCS."""
    jacc = jaccard_similarity(text1, text2)
    lev = levenshtein_similarity(text1, text2)
    lcs = longest_common_subsequence_ratio(text1, text2)

    composite = (0.35 * jacc) + (0.35 * lev) + (0.30 * lcs)
    return round(max(0.0, min(1.0, composite)), 4)


def compute_semantic_congruence(text1: str, text2: str) -> float:
    """Evaluate semantic congruence between original span and LLM prediction."""
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    # Base lexical similarity
    lex = compute_lexical_similarity(t1, t2)

    # Character 3-gram & 4-gram Dice similarity
    def get_char_ngrams(s: str, n: int) -> set:
        padded = f"^{s}$"
        return {padded[i:i+n] for i in range(len(padded) - n + 1)}

    tri_1 = get_char_ngrams(t1, 3)
    tri_2 = get_char_ngrams(t2, 3)
    tri_sim = (2.0 * len(tri_1 & tri_2)) / max(1, len(tri_1) + len(tri_2))

    tetra_1 = get_char_ngrams(t1, 4)
    tetra_2 = get_char_ngrams(t2, 4)
    tetra_sim = (2.0 * len(tetra_1 & tetra_2)) / max(1, len(tetra_1) + len(tetra_2))

    ngram_sim = (0.5 * tri_sim) + (0.5 * tetra_sim)

    # Word prefix / root similarity
    words1 = tokenize_words(t1)
    words2 = tokenize_words(t2)
    stem_matches = 0
    for w1 in words1:
        prefix1 = w1[:4] if len(w1) >= 4 else w1
        if any(w2.startswith(prefix1) or prefix1 in w2 for w2 in words2):
            stem_matches += 1
    stem_ratio = (stem_matches / max(1, len(words1))) if words1 else 0.0

    len1, len2 = max(1, len(t1)), max(1, len(t2))
    len_ratio = min(len1, len2) / max(len1, len2)

    semantic_score = (0.35 * lex) + (0.35 * ngram_sim) + (0.20 * stem_ratio) + (0.10 * len_ratio)

    # Boost shared core words
    set1 = set(words1)
    set2 = set(words2)
    if len(set1) >= 2 and len(set1 & set2) >= max(1, len(set1) // 2):
        semantic_score = max(semantic_score, 0.55)

    return round(max(0.0, min(1.0, semantic_score)), 4)


def calculate_burstiness(text: str) -> Dict[str, float]:
    """Calculate text burstiness and sentence length variance."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
    sentences = [s.strip() for s in sentence_endings.split(text.strip()) if s.strip()]
    
    if len(sentences) <= 1:
        return {"burstiness_score": 0.5, "variance": 0.0, "mean_length": len(text.split())}

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)

    cv = (std_dev / mean_len) if mean_len > 0 else 0.0
    burstiness_score = min(1.0, cv / 0.8)

    return {
        "burstiness_score": round(burstiness_score, 3),
        "sentence_variance": round(variance, 2),
        "mean_sentence_length": round(mean_len, 1),
    }


def classify_span_congruence(congruence_score: float) -> str:
    """Classify a single span's congruence status."""
    if congruence_score >= 0.70:
        return "CONGRUENT"
    elif congruence_score >= 0.40:
        return "PARTIAL"
    else:
        return "DIVERGENT"


def compute_ai_probability(
    span_congruences: List[float],
    lexical_similarities: List[float],
    semantic_similarities: List[float],
    burstiness_score: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive AI Probability percentage and confidence verdict."""
    if not span_congruences:
        return {
            "ai_probability": 50.0,
            "verdict": "Indeterminate (Insufficient Maskable Spans)",
            "confidence": "Low",
            "semantic_similarity_avg": 0.0,
            "lexical_similarity_avg": 0.0,
            "congruence_avg": 0.0,
        }

    avg_congruence = sum(span_congruences) / len(span_congruences)
    avg_lexical = sum(lexical_similarities) / len(lexical_similarities)
    avg_semantic = sum(semantic_similarities) / len(semantic_similarities)

    congruent_spans_count = sum(1 for c in span_congruences if c >= 0.70)
    congruent_ratio = congruent_spans_count / len(span_congruences)

    raw_p = (0.50 * avg_semantic) + (0.30 * avg_lexical) + (0.20 * congruent_ratio)
    burstiness_adjustment = (0.5 - burstiness_score) * 0.18
    adjusted_p = max(0.02, min(0.98, raw_p + burstiness_adjustment))

    k = 7.5
    sigmoid_score = 1.0 / (1.0 + math.exp(-k * (adjusted_p - 0.46)))
    ai_percentage = round(sigmoid_score * 100.0, 1)

    if ai_percentage >= 72.0:
        verdict = "Likely AI-Generated"
        confidence = "High" if ai_percentage >= 85.0 else "Moderate"
    elif ai_percentage >= 45.0:
        verdict = "Mixed / AI-Assisted or Edited"
        confidence = "Moderate"
    else:
        verdict = "Likely Human-Authored"
        confidence = "High" if ai_percentage <= 25.0 else "Moderate"

    return {
        "ai_probability": ai_percentage,
        "verdict": verdict,
        "confidence": confidence,
        "congruence_avg": round(avg_congruence * 100.0, 1),
        "semantic_similarity_avg": round(avg_semantic * 100.0, 1),
        "lexical_similarity_avg": round(avg_lexical * 100.0, 1),
        "congruent_spans_count": congruent_spans_count,
        "total_spans_count": len(span_congruences),
        "congruent_ratio": round(congruent_ratio * 100.0, 1),
    }
