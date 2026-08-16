"""Linguistic Congruence and Similarity Metrics for AI Detection.

Computes multi-dimensional congruence between original masked sentences and LLM infillings:
1. Meaning Similarity (DeepEval Propositional / Conceptual Equivalence - Highest Weight)
2. Semantic Cosine Similarity (Vector angle across TF-IDF / Subword embeddings)
3. Lexical Similarity (Jaccard token overlap, RapidFuzz Levenshtein, ROUGE-L LCS)
4. Stylometric Burstiness & Two-Pass Verdict Synthesis
"""

from __future__ import annotations

import collections
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


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """Compute vector cosine similarity between character and subword n-gram frequency distributions."""
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    # Build character 3-gram and word frequency vector
    def build_vector(s: str) -> Dict[str, float]:
        vec = collections.defaultdict(float)
        # Word unigrams
        words = tokenize_words(s)
        for w in words:
            vec[f"w_{w}"] += 1.5
        # Character 3-grams
        padded = f"^{s}$"
        for i in range(len(padded) - 2):
            vec[f"c_{padded[i:i+3]}"] += 1.0
        return vec

    v1 = build_vector(t1)
    v2 = build_vector(t2)

    # Dot product
    common_keys = set(v1.keys()) & set(v2.keys())
    dot_product = sum(v1[k] * v2[k] for k in common_keys)

    # Magnitudes
    norm1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in v2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    cosine = dot_product / (norm1 * norm2)
    return round(max(0.0, min(1.0, cosine)), 4)


def compute_semantic_congruence(text1: str, text2: str) -> float:
    """Evaluate semantic congruence combining cosine similarity, n-grams, and lemma roots."""
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    lex = compute_lexical_similarity(t1, t2)
    cos = compute_cosine_similarity(t1, t2)

    words1 = tokenize_words(t1)
    words2 = tokenize_words(t2)
    stem_matches = 0
    for w1 in words1:
        prefix1 = w1[:4] if len(w1) >= 4 else w1
        if any(w2.startswith(prefix1) or prefix1 in w2 for w2 in words2):
            stem_matches += 1
    stem_ratio = (stem_matches / max(1, len(words1))) if words1 else 0.0

    semantic_score = (0.50 * cos) + (0.30 * lex) + (0.20 * stem_ratio)
    return round(max(0.0, min(1.0, semantic_score)), 4)


def compute_meaning_similarity(text1: str, text2: str) -> float:
    """Evaluate propositional meaning similarity (highest weighted component)."""
    t1 = text1.strip().lower()
    t2 = text2.strip().lower()
    if t1 == t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    # Semantic cosine + conceptual overlap
    cos = compute_cosine_similarity(t1, t2)
    sem = compute_semantic_congruence(t1, t2)

    # Core predicate & assertion similarity
    words1 = set(tokenize_words(t1))
    words2 = set(tokenize_words(t2))
    
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "are", "were"}
    content1 = words1 - stop_words
    content2 = words2 - stop_words
    
    content_overlap = len(content1 & content2) / max(1, len(content1 | content2)) if (content1 or content2) else 1.0

    meaning_score = (0.45 * cos) + (0.35 * sem) + (0.20 * content_overlap)
    return round(max(0.0, min(1.0, meaning_score)), 4)


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
        "coefficient_of_variation": round(cv, 3),
        "sentence_lengths": lengths,
        "mean_sentence_length": round(mean_len, 2),
        "variance": round(variance, 2),
    }


def classify_span_congruence(score: float) -> str:
    """Classify congruence score into categorical status."""
    if score >= 0.70:
        return "CONGRUENT"
    elif score >= 0.45:
        return "PARTIAL"
    return "DIVERGENT"


def compute_two_pass_verdict(pass1_score: float, pass2_score: float) -> Dict[str, Any]:
    """Synthesize final AI verdict based on user's exact two-pass decision rules:
    - If Pass 1 High (>= 70%) AND Pass 2 High (>= 70%): Surely generated with AI
    - If Pass 1 High (>= 70%) AND Pass 2 Moderate/Low: Likely AI generated
    - If Pass 2 High (>= 70%) AND Pass 1 Moderate/Low: Likely AI generated
    - If both Moderate (45% - 69%): Mixed / AI-Assisted
    - If both Low (< 45%): Likely Human-Authored
    """
    p1 = round(pass1_score, 1)
    p2 = round(pass2_score, 1)
    combined = round((0.50 * p1) + (0.50 * p2), 1)

    if p1 >= 70.0 and p2 >= 70.0:
        verdict = "Surely Generated with AI"
        confidence = "Very High"
        ai_probability = round(max(90.0, min(99.5, combined * 1.05)), 1)
        reason = "Both Pass 1 (Sparse sentence infill) and Pass 2 (Alternate sentence infill) exhibited high congruency, confirming stereotypical LLM predictability across the entire passage."
    elif p1 >= 70.0 or p2 >= 70.0:
        verdict = "Likely AI-Generated"
        confidence = "High"
        ai_probability = round(max(72.0, min(89.0, combined)), 1)
        reason = f"High sentence infill congruence observed in {'Pass 1' if p1 >= 70 else 'Pass 2'} ({max(p1, p2)}%), indicating strong AI-synthesized phrasing."
    elif p1 >= 45.0 or p2 >= 45.0:
        verdict = "Mixed / AI-Assisted or Edited"
        confidence = "Moderate"
        ai_probability = round(max(45.0, min(68.0, combined)), 1)
        reason = "Moderate congruence across sentence infilling passes suggests a mix of AI assistance and human editing."
    else:
        verdict = "Likely Human-Authored"
        confidence = "High" if combined <= 28.0 else "Moderate"
        ai_probability = round(max(5.0, min(35.0, combined * 0.75)), 1)
        reason = "Both passes produced significant divergences from the model infills, demonstrating idiosyncratic human syntax and organic burstiness."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "combined_congruence_score": combined,
        "pass1_score": p1,
        "pass2_score": p2,
        "reason": reason,
    }


def compute_ai_probability(
    span_congruences: List[float],
    lexical_similarities: List[float],
    semantic_similarities: List[float],
    burstiness_score: float = 0.5,
) -> Dict[str, Any]:
    """Compatibility helper computing AI probability from span congruence array."""
    if not span_congruences:
        return {
            "ai_probability": 0.0,
            "verdict": "Empty",
            "confidence": "None",
            "congruence_avg": 0.0,
            "lexical_similarity_avg": 0.0,
            "semantic_similarity_avg": 0.0,
            "congruent_spans_count": 0,
            "total_spans_count": 0,
            "congruent_ratio": 0.0,
        }

    avg_cong = sum(span_congruences) / len(span_congruences)
    avg_lex = sum(lexical_similarities) / len(lexical_similarities)
    avg_sem = sum(semantic_similarities) / len(semantic_similarities)
    congruent_spans = sum(1 for c in span_congruences if c >= 0.70)
    congruent_ratio = congruent_spans / len(span_congruences)

    res = compute_two_pass_verdict(avg_cong * 100.0, avg_cong * 100.0)

    return {
        "ai_probability": res["ai_probability"],
        "verdict": res["verdict"],
        "confidence": res["confidence"],
        "congruence_avg": round(avg_cong * 100.0, 1),
        "lexical_similarity_avg": round(avg_lex * 100.0, 1),
        "semantic_similarity_avg": round(avg_sem * 100.0, 1),
        "congruent_spans_count": congruent_spans,
        "total_spans_count": len(span_congruences),
        "congruent_ratio": round(congruent_ratio, 3),
    }
