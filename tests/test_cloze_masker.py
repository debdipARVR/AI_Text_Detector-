"""Unit tests for Sentence ClozeMasker with [1], [2] Numbered Placeholders and Metadata Extraction."""

from src.engine.cloze_masker import ClozeMasker, ClozeMaskResult, SentenceMetadata


def test_split_sentences():
    text = "Large language models generate text based on statistical likelihood. They optimize for minimal perplexity. However, humans write with idiosyncratic variation."
    sentences = ClozeMasker.split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Large language")
    assert sentences[1].startswith("They optimize")
    assert sentences[2].startswith("However, humans")


def test_extract_sentence_metadata():
    sentence = "Known for his calm personality, intellectual honesty, and understated style of leadership, he served as the Prime Minister of India from 2004 to 2014."
    meta = ClozeMasker.extract_sentence_metadata(sentence)
    
    assert isinstance(meta, SentenceMetadata)
    assert meta.word_count >= 20
    assert meta.space_count >= 20
    assert "," in meta.special_characters
    assert "." in meta.special_characters
    assert meta.punctuation_counts["comma"] >= 3
    assert meta.avg_word_length > 3.0


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
    assert result.spans[0].metadata.word_count >= 8


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
    assert result.spans[0].metadata.word_count > 0
