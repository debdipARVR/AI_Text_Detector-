"""Unit tests for similarity and congruence metrics."""

from src.engine.metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_ai_probability,
    compute_lexical_similarity,
    compute_semantic_congruence,
    jaccard_similarity,
    levenshtein_similarity,
    longest_common_subsequence_ratio,
)


def test_jaccard_similarity():
    # Identical
    assert jaccard_similarity("the quick brown fox", "the quick brown fox") == 1.0
    # Disjoint
    assert jaccard_similarity("the quick brown fox", "completely unrelated statement") == 0.0
    # Overlap
    score = jaccard_similarity("the artificial intelligence model", "the artificial neural model")
    assert 0.4 <= score <= 0.8


def test_levenshtein_similarity():
    assert levenshtein_similarity("identical string", "identical string") == 1.0
    score = levenshtein_similarity("machine learning", "machine learnings")
    assert score > 0.9
    assert levenshtein_similarity("apple", "banana") < 0.4


def test_longest_common_subsequence_ratio():
    assert longest_common_subsequence_ratio("a b c d", "a b c d") == 1.0
    assert longest_common_subsequence_ratio("a b c", "x y z") == 0.0


def test_compute_lexical_similarity():
    score_high = compute_lexical_similarity("crucial implications for safety", "crucial implications for safety")
    assert score_high == 1.0
    score_mid = compute_lexical_similarity("crucial implications for safety", "vital implications regarding safety")
    assert 0.3 <= score_mid <= 0.8


def test_compute_semantic_congruence():
    exact = compute_semantic_congruence("delving into the landscape", "delving into the landscape")
    assert exact == 1.0
    similar = compute_semantic_congruence("deeply exploring the landscape", "delving into the landscape")
    assert similar >= 0.5


def test_calculate_burstiness():
    # Uniform text (typical AI)
    uniform = "This is a sentence. This is another sentence. This is a third sentence. This is a fourth sentence."
    burst_uniform = calculate_burstiness(uniform)
    
    # Highly spiky text (typical Human)
    spiky = "Yes. In the vast and intricate labyrinth of human consciousness, we often stumble upon profound epiphanies that defy reductionist logic. Why? Because."
    burst_spiky = calculate_burstiness(spiky)
    
    assert burst_spiky["burstiness_score"] >= burst_uniform["burstiness_score"]


def test_compute_ai_probability():
    # High congruence -> High AI probability
    high_spans = [0.95, 0.90, 0.88]
    res_high = compute_ai_probability(high_spans, [0.90, 0.85, 0.85], [0.95, 0.92, 0.90], burstiness_score=0.2)
    assert res_high["ai_probability"] >= 70.0
    assert "AI" in res_high["verdict"]

    # Low congruence -> Low AI probability (Human)
    low_spans = [0.15, 0.25, 0.10]
    res_low = compute_ai_probability(low_spans, [0.20, 0.20, 0.15], [0.25, 0.30, 0.18], burstiness_score=0.8)
    assert res_low["ai_probability"] <= 35.0
    assert "Human" in res_low["verdict"]


def test_classify_span_congruence():
    assert classify_span_congruence(0.85) == "CONGRUENT"
    assert classify_span_congruence(0.55) == "PARTIAL"
    assert classify_span_congruence(0.20) == "DIVERGENT"
