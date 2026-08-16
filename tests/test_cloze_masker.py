"""Unit tests for ClozeMasker Pass 2 (Alternate) and Pass 3 (Middle 3-Sentence per Passage)."""

from src.engine.cloze_masker import ClozeMasker, ClozeMaskResult


def test_split_into_paragraphs():
    text = (
        "Paragraph one sentence one. Paragraph one sentence two.\n\n"
        "Paragraph two sentence one. Paragraph two sentence two. Paragraph two sentence three."
    )
    paragraphs = ClozeMasker.split_into_paragraphs(text)
    assert len(paragraphs) == 2
    assert len(paragraphs[0]) == 2
    assert len(paragraphs[1]) == 3


def test_mask_pass_2_alternate():
    masker = ClozeMasker()
    text = (
        "Sentence zero kept as context. "
        "Sentence one removed in alternate pass. "
        "Sentence two kept as context. "
        "Sentence three removed in alternate pass."
    )
    result = masker.mask_pass_2(text)
    assert isinstance(result, ClozeMaskResult)
    assert result.pass_index == 2
    assert len(result.spans) == 2
    assert "[1]" in result.masked_text
    assert "[2]" in result.masked_text


def test_mask_pass_3_middle_sentences():
    masker = ClozeMasker()
    # 5-sentence paragraph: should remove 3 middle sentences (1, 2, 3) keeping 0 and 4
    p1 = "S0 opening topic sentence. S1 first middle sentence. S2 center sentence. S3 third middle sentence. S4 concluding sentence."
    # Second paragraph: 5 sentences
    p2 = "P2S0 opening topic. P2S1 first middle. P2S2 center. P2S3 third middle. P2S4 conclusion."
    essay = f"{p1}\n\n{p2}"

    result = masker.mask_pass_3(essay)
    assert isinstance(result, ClozeMaskResult)
    assert result.pass_index == 3
    assert result.total_paragraphs == 2
    assert len(result.spans) == 6  # 3 from p1 + 3 from p2
    assert "[1]" in result.masked_text
    assert "[6]" in result.masked_text

    # Verify that S0 and S4 are kept as context anchors in p1
    assert "S0 opening topic sentence." in result.masked_text
    assert "S4 concluding sentence." in result.masked_text
