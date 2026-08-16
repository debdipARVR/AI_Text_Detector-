"""Unit tests for ClozeMasker."""

from src.engine.cloze_masker import ClozeMasker, ClozeMaskResult


def test_split_sentences():
    text = "Large language models generate text based on statistical likelihood. They optimize for minimal perplexity. However, humans write with idiosyncratic variation."
    sentences = ClozeMasker.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Large language")
    assert sentences[1].startswith("They optimize")
    assert sentences[2].startswith("However, humans")


def test_mask_text_basic():
    masker = ClozeMasker(default_mask_rate=0.30, seed=42)
    text = (
        "Artificial intelligence systems are rapidly advancing across multimodal domains. "
        "These models demonstrate emergent reasoning abilities that surpass traditional heuristics. "
        "Understanding how they formulate conclusions remains an active area of contemporary research."
    )
    result = masker.mask_text(text)
    assert isinstance(result, ClozeMaskResult)
    assert len(result.spans) > 0
    assert "[MASK_1]" in result.masked_text
    assert result.masked_words > 0
    assert result.total_words > 20
    assert 0.10 <= result.mask_ratio <= 0.60


def test_multipass_masks():
    masker = ClozeMasker(default_mask_rate=0.30)
    text = (
        "Neural networks learn representations from high dimensional data distributions. "
        "By optimizing gradient descent on billions of parameters, they capture nuanced patterns."
    )
    passes = masker.generate_multipass_masks(text, num_passes=3)
    assert len(passes) == 3
    for p in passes:
        assert isinstance(p, ClozeMaskResult)
        assert len(p.spans) >= 1
