"""Sentence-Level Cloze Masking Engine with Stylometric & Structural Metadata Extraction.

Extracts complete sentences, replaces them with numbered placeholders [1], [2], [3]...,
and calculates comprehensive structural metadata for each removed sentence:
- Exact Word Count
- Space Count and Spacing Frequency
- Special Characters (commas, semicolons, dashes, quotes, parentheses)
- Punctuation Distribution Profile
- Average Word Length & Capitalization Metrics
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SentenceMetadata:
    """Detailed structural and stylometric metadata for a removed sentence."""
    word_count: int
    char_count: int
    space_count: int
    special_characters: List[str] = field(default_factory=list)
    punctuation_counts: Dict[str, int] = field(default_factory=dict)
    avg_word_length: float = 0.0
    leading_capitalized: bool = True
    has_proper_nouns: bool = False

    def to_prompt_constraint(self) -> str:
        """Format metadata as an explicit prompt constraint for LLM completion."""
        spec_chars_str = ", ".join(f"'{c}'" for c in sorted(set(self.special_characters))) if self.special_characters else "None"
        punct_desc = ", ".join(f"{k}: {v}" for k, v in self.punctuation_counts.items() if v > 0)
        return (
            f"Target Word Count: exactly {self.word_count} words | "
            f"Space Count: {self.space_count} | "
            f"Special Characters: [{spec_chars_str}] | "
            f"Punctuation Profile: ({punct_desc or 'Standard period ending'}) | "
            f"Avg Word Length: ~{self.avg_word_length} chars"
        )


@dataclass
class MaskedSpan:
    """Represents a masked complete sentence span with structural metadata."""
    mask_id: int
    placeholder: str
    original_text: str
    sentence_idx: int
    metadata: SentenceMetadata = field(default_factory=lambda: SentenceMetadata(0, 0, 0))
    char_start: int = 0
    char_end: int = 0
    context_prefix: str = ""
    context_suffix: str = ""
    predicted_text: Optional[str] = None
    infilled_word_count: int = 0
    word_count_delta: int = 0
    meaning_similarity: float = 0.0
    cosine_similarity: float = 0.0
    semantic_similarity: float = 0.0
    lexical_similarity: float = 0.0
    composite_congruence: float = 0.0
    status: str = "PENDING"  # PENDING, CONGRUENT, PARTIAL, DIVERGENT


@dataclass
class ClozeMaskResult:
    """Result of sentence masking for Cloze infilling."""
    original_text: str
    masked_text: str
    spans: List[MaskedSpan] = field(default_factory=list)
    total_sentences: int = 0
    masked_sentences_count: int = 0
    total_words: int = 0
    masked_words: int = 0
    mask_ratio: float = 0.0
    pass_index: int = 1
    pass_name: str = "Pass 1 (Sparse - 1 sentence per 4 lines)"


class ClozeMasker:
    """Masks complete sentences into [1], [2], [3] placeholder keys with structural metadata."""

    def __init__(self, default_mask_rate: float = 0.30):
        self.default_mask_rate = default_mask_rate

    @staticmethod
    def extract_sentence_metadata(sentence: str) -> SentenceMetadata:
        """Extract exact word count, space frequency, and special characters from a sentence."""
        words = re.findall(r'\b[a-zA-Z0-9_\'-]+\b', sentence)
        word_count = len(words)
        char_count = len(sentence)
        space_count = sentence.count(" ") + sentence.count("\t")

        # Find all special characters (non-alphanumeric, non-space)
        special_chars = [c for c in sentence if not c.isalnum() and not c.isspace()]

        # Punctuation counts breakdown
        punct_map = {
            "comma": sentence.count(","),
            "period": sentence.count("."),
            "semicolon": sentence.count(";"),
            "colon": sentence.count(":"),
            "dash": sentence.count("-") + sentence.count("—") + sentence.count("–"),
            "quotes": sentence.count('"') + sentence.count("'"),
            "parentheses": sentence.count("(") + sentence.count(")"),
        }

        total_word_len = sum(len(w) for w in words)
        avg_len = round(total_word_len / max(1, word_count), 2)
        has_prop = any(w[0].isupper() for w in words[1:] if len(w) > 1) if word_count > 1 else False

        return SentenceMetadata(
            word_count=word_count,
            char_count=char_count,
            space_count=space_count,
            special_characters=special_chars,
            punctuation_counts=punct_map,
            avg_word_length=avg_len,
            leading_capitalized=sentence[0].isupper() if sentence else True,
            has_proper_nouns=has_prop,
        )

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text cleanly into sentences while filtering out empty lines, markdown headers, and Roman numerals."""
        cleaned = text.strip()
        if not cleaned:
            return []

        raw_chunks = re.split(r'\n{2,}|\n(?=[A-Z0-9#])', cleaned)
        sentences: List[str] = []

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Skip markdown headers or standalone numbers / Roman numerals
            if chunk.startswith("#") and len(chunk.split()) < 8:
                continue
            if re.match(r'^(?:[IVXLCDM]+\.?|[0-9]+\.?)$', chunk, re.IGNORECASE):
                continue

            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', chunk)
            for p in parts:
                s = p.strip()
                if len(s.split()) >= 4 or (len(s.split()) >= 2 and any(w in s.lower() for w in ["is", "was", "are", "were", "has", "had", "became", "served", "born", "died", "held", "faced", "studied"])):
                    sentences.append(s)

        if not sentences and cleaned:
            sentences = [cleaned]

        return sentences

    def mask_pass_1(self, text: str) -> ClozeMaskResult:
        """Pass 1: Sparse complete sentence removal (1 sentence removed per 4 sentences)."""
        sentences = self.split_into_sentences(text)
        n = len(sentences)
        if n == 0:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=1)

        masked_indices: List[int] = []
        if n == 1:
            masked_indices = [0]
        elif n <= 3:
            masked_indices = [1 if n > 1 else 0]
        else:
            for i in range(1, n, 4):
                masked_indices.append(i)
            if not masked_indices:
                masked_indices = [1]

        return self._build_masked_result(
            sentences=sentences,
            masked_indices=masked_indices,
            pass_index=1,
            pass_name="Pass 1 (Sparse: 1 sentence per 4 lines)",
            original_text=text,
        )

    def mask_pass_2(self, text: str) -> ClozeMaskResult:
        """Pass 2: Alternate complete sentence removal (sentences removed every 2 sentences)."""
        sentences = self.split_into_sentences(text)
        n = len(sentences)
        if n == 0:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=2)

        masked_indices: List[int] = []
        if n == 1:
            masked_indices = [0]
        elif n == 2:
            masked_indices = [0]
        else:
            for i in range(1, n, 2):
                masked_indices.append(i)
            if not masked_indices:
                masked_indices = [0]

        return self._build_masked_result(
            sentences=sentences,
            masked_indices=masked_indices,
            pass_index=2,
            pass_name="Pass 2 (Alternate: sentences removed every 2 lines)",
            original_text=text,
        )

    def _build_masked_result(
        self,
        sentences: List[str],
        masked_indices: List[int],
        pass_index: int,
        pass_name: str,
        original_text: str,
    ) -> ClozeMaskResult:
        spans: List[MaskedSpan] = []
        masked_sentences: List[str] = list(sentences)
        mask_counter = 1

        total_words = sum(len(s.split()) for s in sentences)
        masked_words = 0

        for idx in masked_indices:
            if idx >= len(sentences):
                continue
            orig_s = sentences[idx]
            placeholder = f"[{mask_counter}]"
            masked_sentences[idx] = placeholder
            
            meta = self.extract_sentence_metadata(orig_s)
            masked_words += meta.word_count

            prefix = sentences[idx - 1] if idx > 0 else ""
            suffix = sentences[idx + 1] if idx + 1 < len(sentences) else ""

            spans.append(
                MaskedSpan(
                    mask_id=mask_counter,
                    placeholder=placeholder,
                    original_text=orig_s,
                    sentence_idx=idx,
                    metadata=meta,
                    context_prefix=prefix,
                    context_suffix=suffix,
                )
            )
            mask_counter += 1

        masked_text = " ".join(masked_sentences)
        ratio = round((masked_words / max(1, total_words)), 3)

        return ClozeMaskResult(
            original_text=original_text,
            masked_text=masked_text,
            spans=spans,
            total_sentences=len(sentences),
            masked_sentences_count=len(spans),
            total_words=total_words,
            masked_words=masked_words,
            mask_ratio=ratio,
            pass_index=pass_index,
            pass_name=pass_name,
        )

    def mask_text(
        self,
        text: str,
        mask_rate: Optional[float] = None,
        pass_index: int = 1,
    ) -> ClozeMaskResult:
        if pass_index == 2:
            return self.mask_pass_2(text)
        return self.mask_pass_1(text)
