"""Sentence-Level Cloze Masking Engine for Two-Pass AI Text Detection.

Replaces random word masking with complete sentence extraction:
- Pass 1 (Sparse): Removes 1 sentence every 4 sentences/lines.
- Pass 2 (Alternate / Dense): Removes alternate sentences every 2 sentences/lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MaskedSpan:
    """Represents a masked complete sentence span within a paragraph."""
    mask_id: int
    placeholder: str
    original_text: str
    sentence_idx: int
    char_start: int = 0
    char_end: int = 0
    context_prefix: str = ""
    context_suffix: str = ""
    predicted_text: Optional[str] = None
    meaning_similarity: float = 0.0
    semantic_similarity: float = 0.0
    cosine_similarity: float = 0.0
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
    pass_name: str = "Pass 1 (Sparse - 1 per 4 sentences)"


class ClozeMasker:
    """Masks complete sentences in structured passes (1 per 4 sentences vs alternate every 2 sentences)."""

    def __init__(self, default_mask_rate: float = 0.30):
        self.default_mask_rate = default_mask_rate

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text cleanly into sentences while preserving sentence boundaries and punctuation."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # Split on sentence terminals (.!?\n) followed by whitespace or capital letters
        pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|(?<=\n)\s*'
        raw_parts = re.split(pattern, cleaned)
        sentences: List[str] = []

        for p in raw_parts:
            s = p.strip()
            if len(s) > 1:
                sentences.append(s)

        # Fallback if no clear punctuation breaks
        if not sentences and cleaned:
            # Split by lines or commas if long
            lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
            if lines:
                return lines
            sentences = [cleaned]

        return sentences

    def mask_pass_1(self, text: str) -> ClozeMaskResult:
        """Pass 1: Sparse complete sentence removal.
        Removes 1 sentence for every 4 sentences/lines (e.g. index 1, 5, etc. or index 0 for short text).
        """
        sentences = self.split_into_sentences(text)
        n = len(sentences)
        if n == 0:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=1)

        masked_indices: List[int] = []
        if n == 1:
            masked_indices = [0]
        elif n <= 3:
            # For 2-3 sentences, mask 1 middle sentence
            masked_indices = [1 if n > 1 else 0]
        else:
            # 1 sentence removed for every 4 sentences (indices 1, 5, 9...)
            for i in range(1, n, 4):
                masked_indices.append(i)
            # Ensure at least one sentence is masked
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
        """Pass 2: Alternate sentence removal.
        Removes alternate sentences every 2 sentences/lines (e.g., indices 0, 2, 4... or 1, 3...).
        """
        sentences = self.split_into_sentences(text)
        n = len(sentences)
        if n == 0:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=2)

        masked_indices: List[int] = []
        if n == 1:
            masked_indices = [0]
        elif n == 2:
            # Mask both or 1st for 2 sentences
            masked_indices = [0, 1] if len(sentences[0].split()) < 15 else [0]
        else:
            # Alternate sentences removed every 2 sentences (e.g. 0, 2, 4... or 1, 3...)
            # For standard paragraphs, mask indices 0 and 2 (or 1 and 3)
            for i in range(0, n, 2):
                masked_indices.append(i)

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
            placeholder = f"[MASK_{mask_counter}]"
            masked_sentences[idx] = placeholder
            
            w_count = len(orig_s.split())
            masked_words += w_count

            prefix = sentences[idx - 1] if idx > 0 else ""
            suffix = sentences[idx + 1] if idx + 1 < len(sentences) else ""

            spans.append(
                MaskedSpan(
                    mask_id=mask_counter,
                    placeholder=placeholder,
                    original_text=orig_s,
                    sentence_idx=idx,
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
        """Convenience method to execute Pass 1 or Pass 2 based on pass_index."""
        if pass_index == 2:
            return self.mask_pass_2(text)
        return self.mask_pass_1(text)
