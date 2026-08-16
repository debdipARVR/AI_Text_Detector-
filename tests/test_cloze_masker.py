"""Unit tests for Sentence ClozeMasker with [1], [2] Numbered Placeholders."""

from src.engine.cloze_masker import ClozeMasker, ClozeMaskResult


def test_split_sentences():
    text = "Large language models generate text based on statistical likelihood. They optimize for minimal perplexity. However, humans write with idiosyncratic variation."
    sentences = ClozeMasker.split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Large language")
    assert sentences[1].startswith("They optimize")
    assert sentences[2].startswith("However, humans")


def test_mask_pass_1_sparse():
    masker = ClozeMasker()
    text = (
        "First sentence in the paragraph. "
        "Second sentence that should be masked in sparse mode. "
        "Third sentence following the masked sentence. "
        "Fourth sentence ending the first four line group. "
        "Fifth sentence starting the next group."
    )
    result = masker.mask_pass_1(text)
    assert isinstance(result, ClozeMaskResult)
    assert result.pass_index == 1
    assert len(result.spans) >= 1
    assert "[1]" in result.masked_text
    assert result.spans[0].original_text == "Second sentence that should be masked in sparse mode."


def test_mask_pass_2_alternate():
    masker = ClozeMasker()
    text = (
        "Sentence zero removed in alternate pass. "
        "Sentence one kept as context. "
        "Sentence two removed in alternate pass. "
        "Sentence three kept as context."
    )
    result = masker.mask_pass_2(text)
    assert isinstance(result, ClozeMaskResult)
    assert result.pass_index == 2
    assert len(result.spans) == 2
    assert "[1]" in result.masked_text
    assert "[2]" in result.masked_text
    assert result.spans[0].placeholder == "[1]"
    assert result.spans[1].placeholder == "[2]"
