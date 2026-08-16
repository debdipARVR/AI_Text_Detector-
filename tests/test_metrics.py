"""Unit tests for similarity and congruence metrics."""

from src.engine.metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_cosine_similarity,
    compute_lexical_similarity,
    compute_meaning_similarity,
    compute_semantic_congruence,
    compute_two_pass_verdict,
    jaccard_similarity,
    levenshtein_similarity,
    longest_common_subsequence_ratio,
)


def test_jaccard_similarity():
    assert jaccard_similarity("the quick brown fox", "the quick brown fox") == 1.0
    assert jaccard_similarity("the quick brown fox", "completely unrelated statement") == 0.0
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


def test_compute_cosine_similarity():
    exact = compute_cosine_similarity("Large language models are transforming computing.", "Large language models are transforming computing.")
    assert exact == 1.0
    similar = compute_cosine_similarity("Large language models transform computational domains.", "Language models are transforming computing.")
    assert similar >= 0.5


def test_compute_meaning_similarity():
    exact = compute_meaning_similarity("AI has revolutionized modern technology paradigms.", "AI has revolutionized modern technology paradigms.")
    assert exact == 1.0
    similar = compute_meaning_similarity("AI has fundamentally shifted technological systems.", "AI has revolutionized modern technology paradigms.")
    assert similar >= 0.25


def test_compute_two_pass_verdict():
    # Pass 1 High & Pass 2 High -> Surely AI
    res_sure = compute_two_pass_verdict(85.0, 80.0)
    assert res_sure["verdict"] == "Surely Generated with AI"
    assert res_sure["ai_probability"] >= 90.0

    # Pass 1 High & Pass 2 Moderate -> Likely AI
    res_likely1 = compute_two_pass_verdict(80.0, 30.0)
    assert res_likely1["verdict"] == "Likely AI-Generated"
    assert res_likely1["ai_probability"] >= 70.0

    # Pass 2 High & Pass 1 Moderate -> Likely AI
    res_likely2 = compute_two_pass_verdict(30.0, 80.0)
    assert res_likely2["verdict"] == "Likely AI-Generated"
    assert res_likely2["ai_probability"] >= 70.0

    # Both Low -> Likely Human
    res_human = compute_two_pass_verdict(20.0, 15.0)
    assert res_human["verdict"] == "Likely Human-Authored"
    assert res_human["ai_probability"] <= 35.0


def test_classify_span_congruence():
    assert classify_span_congruence(0.85) == "CONGRUENT"
    assert classify_span_congruence(0.55) == "PARTIAL"
    assert classify_span_congruence(0.20) == "DIVERGENT"
