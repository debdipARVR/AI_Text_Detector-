"""Sentence-Level Cloze Masking Engine for Two-Pass AI Text Detection.

Extracts complete sentences and replaces them with numbered placeholders [1], [2], [3]...
Filters out headings, bullet points, Roman numerals, and short titles.
- Pass 1 (Sparse): Removes 1 sentence every 4 sentences.
- Pass 2 (Alternate): Removes alternate sentences every 2 sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MaskedSpan:
    """Represents a masked complete sentence span with a numbered key [1], [2], etc."""
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
    """Masks complete sentences into [1], [2], [3] placeholder keys."""

    def __init__(self, default_mask_rate: float = 0.30):
        self.default_mask_rate = default_mask_rate

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text cleanly into sentences while filtering out empty lines, markdown headers, and Roman numerals."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # Split on paragraph breaks and sentence boundary punctuation followed by spaces or newlines
        raw_chunks = re.split(r'\n{2,}|\n(?=[A-Z0-9#])', cleaned)
        sentences: List[str] = []

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Skip standalone markdown headers (e.g. '# Title') or Roman numerals (e.g. 'V.')
            if chunk.startswith("#") and len(chunk.split()) < 8:
                continue
            if re.match(r'^(?:[IVXLCDM]+\.?|[0-9]+\.?)$', chunk, re.IGNORECASE):
                continue

            # Split sentences within chunk by terminal punctuation (.!?), taking care of abbreviations
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', chunk)
            for p in parts:
                s = p.strip()
                # Exclude trivial non-sentences (< 4 words and no verb structure)
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
            # 1 sentence removed for every 4 sentences (e.g. index 1, 5, 9...)
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
            # Alternate sentences removed every 2 sentences (e.g. 1, 3, 5... or 0, 2, 4...)
            # For essays, masking indices 1, 3, 5... leaves leading topic sentences as context
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
            # Form standard numbered placeholder [1], [2], [3]
            placeholder = f"[{mask_counter}]"
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
        """Convenience method for Pass 1 or Pass 2."""
        if pass_index == 2:
            return self.mask_pass_2(text)
        return self.mask_pass_1(text)
