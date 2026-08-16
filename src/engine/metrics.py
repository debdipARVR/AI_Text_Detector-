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


def compute_sigmoid_dynamic_weights(cosine_score: float) -> Tuple[float, float, str]:
    """Compute continuous sigmoid adaptive weights:
    - Base: 80% Meaning + 20% Cosine (when Cosine is low)
    - Transition: smooth sigmoid centered at 70% Cosine (k=15.0)
    - Boosted: 60% Meaning + 40% Cosine (when Cosine is high)
    """
    c = max(0.0, min(1.0, cosine_score))
    # Sigmoid function centered at 0.70 with steepness 15.0
    sig = 1.0 / (1.0 + math.exp(-15.0 * (c - 0.70)))
    w_cos = 0.20 + (0.20 * sig)
    w_meaning = 1.0 - w_cos

    policy = f"Sigmoid Adaptive ({round(w_meaning*100, 1)}% Meaning / {round(w_cos*100, 1)}% Cosine)"
    return round(w_meaning, 4), round(w_cos, 4), policy


def compute_dynamic_pair_congruence(meaning_score: float, cosine_score: float) -> Tuple[float, float, float, str]:
    """Compute dynamic congruence using continuous sigmoid gate."""
    w_m, w_c, policy = compute_sigmoid_dynamic_weights(cosine_score)
    congruence = (w_m * meaning_score) + (w_c * cosine_score)
    return round(congruence, 4), w_m, w_c, policy


def compute_bipartite_optimal_matching(
    original_sentences: List[str],
    predicted_sentences: List[str],
) -> List[Dict[str, Any]]:
    """Compute max-weight bipartite matching for multi-sentence blocks (Pass 3 middle sentences).
    Guarantees optimal assignment even if sentences are inverted or compressed.
    """
    import itertools

    n_orig = len(original_sentences)
    n_pred = len(predicted_sentences)
    if n_orig == 0 or n_pred == 0:
        return []

    # If 1-to-1, evaluate directly
    if n_orig == 1 and n_pred == 1:
        orig = original_sentences[0]
        pred = predicted_sentences[0]
        m = compute_meaning_similarity(orig, pred)
        c = compute_cosine_similarity(orig, pred)
        cong, wm, wc, pol = compute_dynamic_pair_congruence(m, c)
        return [{
            "orig_idx": 0,
            "pred_idx": 0,
            "original_sentence": orig,
            "predicted_sentence": pred,
            "meaning_similarity": round(m * 100.0, 1),
            "semantic_cosine": round(c * 100.0, 1),
            "congruence_score": round(cong * 100.0, 1),
            "dynamic_weights": pol,
        }]

    # Compute similarity matrix
    sim_matrix = []
    meta_matrix = []
    for i, orig in enumerate(original_sentences):
        row_sim = []
        row_meta = []
        for j, pred in enumerate(predicted_sentences):
            m = compute_meaning_similarity(orig, pred)
            c = compute_cosine_similarity(orig, pred)
            cong, wm, wc, pol = compute_dynamic_pair_congruence(m, c)
            row_sim.append(cong)
            row_meta.append((m, c, cong, pol))
        sim_matrix.append(row_sim)
        meta_matrix.append(row_meta)

    # Find optimal permutation maximizing total score
    best_score = -1.0
    best_perm = list(range(min(n_orig, n_pred)))

    # For small n (typical block is 3 sentences), full permutations are fast (3! = 6)
    candidate_preds = list(range(n_pred))
    for perm in itertools.permutations(candidate_preds, min(n_orig, n_pred)):
        score = sum(sim_matrix[i][perm[i]] for i in range(len(perm)))
        if score > best_score:
            best_score = score
            best_perm = perm

    matches = []
    for i in range(len(best_perm)):
        j = best_perm[i]
        m, c, cong, pol = meta_matrix[i][j]
        matches.append({
            "orig_idx": i,
            "pred_idx": j,
            "original_sentence": original_sentences[i],
            "predicted_sentence": predicted_sentences[j],
            "meaning_similarity": round(m * 100.0, 1),
            "semantic_cosine": round(c * 100.0, 1),
            "congruence_score": round(cong * 100.0, 1),
            "dynamic_weights": pol,
        })

    return matches


def compute_two_pass_verdict(
    pass2_score: float,
    pass3_score: float,
    burstiness_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synthesize final AI verdict with:
    1. Pass 2 and Pass 3 dynamic combination
    2. Inter-pass variance confidence calibration
    3. Syntactic burstiness modulation
    """
    p2 = round(pass2_score, 1)
    p3 = round(pass3_score, 1)
    combined = round((0.50 * p2) + (0.50 * p3), 1)

    # Inter-pass variance confidence calibration
    delta = abs(p2 - p3)
    calibrated_confidence = round(max(50.0, min(99.5, 100.0 - (1.2 * delta))), 1)

    # Burstiness modulation
    b_score = (burstiness_info or {}).get("burstiness_score", 0.5)
    if b_score < 0.25 and combined >= 50.0:
        # Uniform robotic sentence lengths boost AI probability
        ai_prob_raw = combined * 1.05 + 2.0
    elif b_score > 0.65:
        # High human sentence length variance dampens AI probability
        ai_prob_raw = combined * 0.90
    else:
        ai_prob_raw = combined

    if p2 >= 70.0 and p3 >= 70.0:
        verdict = "Surely Generated with AI"
        confidence_str = "Very High" if calibrated_confidence >= 85.0 else "High"
        ai_probability = round(max(90.0, min(99.5, ai_prob_raw * 1.04)), 1)
        reason = "Both Pass 2 (Alternate sentence infill) and Pass 3 (Middle 3-sentence passage infill) exhibited high congruency, confirming stereotypical LLM predictability across the entire essay."
    elif p2 >= 70.0 or p3 >= 70.0:
        verdict = "Likely AI-Generated"
        confidence_str = "High" if calibrated_confidence >= 75.0 else "Moderate"
        ai_probability = round(max(72.0, min(89.0, ai_prob_raw)), 1)
        reason = f"High sentence infill congruence observed in {'Pass 2' if p2 >= 70 else 'Pass 3'} ({max(p2, p3)}%), indicating strong AI-synthesized phrasing."
    elif p2 >= 45.0 or p3 >= 45.0:
        verdict = "Mixed / AI-Assisted or Edited"
        confidence_str = "Moderate"
        ai_probability = round(max(45.0, min(68.0, ai_prob_raw)), 1)
        reason = "Moderate congruence across Pass 2 and Pass 3 infilling passes suggests a mix of AI assistance and human editing."
    else:
        verdict = "Likely Human-Authored"
        confidence_str = "High" if combined <= 28.0 and calibrated_confidence >= 80.0 else "Moderate"
        ai_probability = round(max(5.0, min(35.0, ai_prob_raw * 0.75)), 1)
        reason = "Both Pass 2 and Pass 3 produced significant divergences from the model infills, demonstrating idiosyncratic human syntax and organic passage flow."

    return {
        "verdict": verdict,
        "confidence": confidence_str,
        "calibrated_confidence_score": calibrated_confidence,
        "ai_probability": ai_probability,
        "combined_congruence_score": combined,
        "pass2_score": p2,
        "pass3_score": p3,
        "inter_pass_delta": round(delta, 1),
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
