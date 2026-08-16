"""Cloze Masking Engine for Randomized Sentence and Clause Masking.

Provides sentence-level and clause-level text masking for Cloze Congruence
AI detection.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MaskedSpan:
    """Represents a masked span within a text document."""
    mask_id: int
    placeholder: str
    original_text: str
    sentence_idx: int
    char_start: int
    char_end: int
    context_prefix: str = ""
    context_suffix: str = ""
    predicted_text: Optional[str] = None
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    composite_congruence: float = 0.0
    status: str = "PENDING"  # PENDING, CONGRUENT, PARTIAL, DIVERGENT


@dataclass
class ClozeMaskResult:
    """Result of masking a document for Cloze infilling."""
    original_text: str
    masked_text: str
    spans: List[MaskedSpan] = field(default_factory=list)
    total_words: int = 0
    masked_words: int = 0
    mask_ratio: float = 0.0
    pass_index: int = 0


class ClozeMasker:
    """Masks random sentences or sub-sentence clauses to evaluate LLM infill congruence."""

    def __init__(
        self,
        default_mask_rate: float = 0.30,
        min_span_words: int = 3,
        max_span_words: int = 14,
        seed: Optional[int] = None,
    ):
        self.default_mask_rate = max(0.10, min(0.60, default_mask_rate))
        self.min_span_words = min_span_words
        self.max_span_words = max_span_words
        self.rng = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences while preserving standard punctuation boundaries."""
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        raw_sentences = sentence_endings.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def mask_text(
        self,
        text: str,
        mask_rate: Optional[float] = None,
        strategy: str = "adaptive",  # adaptive, clause, sentence
        pass_index: int = 0,
    ) -> ClozeMaskResult:
        """Create a masked version of input text replacing selected spans with placeholders."""
        rate = mask_rate if mask_rate is not None else self.default_mask_rate
        rate = max(0.10, min(0.60, rate))
        
        sentences = self.split_sentences(text)
        total_words = len(re.findall(r'\b\w+\b', text))
        target_masked_words = max(3, int(total_words * rate))
        
        spans: List[MaskedSpan] = []
        masked_sentence_list: List[str] = []
        current_masked_words = 0
        mask_counter = 1

        is_short = len(sentences) <= 2

        for s_idx, sentence in enumerate(sentences):
            words = sentence.split()
            n_words = len(words)

            if n_words < 4:
                masked_sentence_list.append(sentence)
                continue

            should_mask = (current_masked_words < target_masked_words) and (self.rng.random() < 0.90 or mask_counter == 1)

            if should_mask and current_masked_words < target_masked_words:
                if strategy == "sentence" and not is_short and n_words <= 25 and self.rng.random() < 0.4:
                    placeholder = f"[MASK_{mask_counter}]"
                    span = MaskedSpan(
                        mask_id=mask_counter,
                        placeholder=placeholder,
                        original_text=sentence,
                        sentence_idx=s_idx,
                        char_start=0,
                        char_end=len(sentence),
                    )
                    spans.append(span)
                    masked_sentence_list.append(placeholder)
                    current_masked_words += n_words
                    mask_counter += 1
                else:
                    span_len = max(
                        self.min_span_words,
                        min(
                            self.max_span_words,
                            int(n_words * self.rng.uniform(0.35, 0.60))
                        )
                    )
                    span_len = min(span_len, n_words - 1)
                    if span_len < 2:
                        span_len = min(2, n_words)

                    max_start = max(0, n_words - span_len)
                    start_idx = self.rng.randint(0, max_start)
                    end_idx = start_idx + span_len

                    masked_words_slice = words[start_idx:end_idx]
                    masked_span_str = " ".join(masked_words_slice)
                    placeholder = f"[MASK_{mask_counter}]"

                    reconstructed_sentence = (
                        " ".join(words[:start_idx]) + (" " if start_idx > 0 else "") +
                        placeholder +
                        (" " if end_idx < n_words else "") + " ".join(words[end_idx:])
                    ).strip()

                    span = MaskedSpan(
                        mask_id=mask_counter,
                        placeholder=placeholder,
                        original_text=masked_span_str,
                        sentence_idx=s_idx,
                        char_start=0,
                        char_end=len(masked_span_str),
                        context_prefix=" ".join(words[:start_idx]),
                        context_suffix=" ".join(words[end_idx:]),
                    )
                    spans.append(span)
                    masked_sentence_list.append(reconstructed_sentence)
                    current_masked_words += len(masked_words_slice)
                    mask_counter += 1
            else:
                masked_sentence_list.append(sentence)

        # Fallback guarantee: if no spans were masked but text has enough words, mask one span from the longest sentence
        if not spans and sentences:
            longest_idx = max(range(len(sentences)), key=lambda i: len(sentences[i].split()))
            s_words = sentences[longest_idx].split()
            if len(s_words) >= 3:
                span_len = min(len(s_words), max(2, len(s_words) // 2))
                start_idx = 0
                end_idx = span_len
                masked_words_slice = s_words[start_idx:end_idx]
                masked_span_str = " ".join(masked_words_slice)
                placeholder = "[MASK_1]"
                
                reconstructed_sentence = (
                    placeholder + (" " if end_idx < len(s_words) else "") + " ".join(s_words[end_idx:])
                ).strip()
                
                span = MaskedSpan(
                    mask_id=1,
                    placeholder=placeholder,
                    original_text=masked_span_str,
                    sentence_idx=longest_idx,
                    char_start=0,
                    char_end=len(masked_span_str),
                    context_prefix="",
                    context_suffix=" ".join(s_words[end_idx:]),
                )
                spans.append(span)
                masked_sentence_list[longest_idx] = reconstructed_sentence
                current_masked_words = len(masked_words_slice)

        masked_text = " ".join(masked_sentence_list)
        actual_ratio = (current_masked_words / total_words) if total_words > 0 else 0.0

        return ClozeMaskResult(
            original_text=text,
            masked_text=masked_text,
            spans=spans,
            total_words=total_words,
            masked_words=current_masked_words,
            mask_ratio=round(actual_ratio, 3),
            pass_index=pass_index,
        )

    def generate_multipass_masks(
        self,
        text: str,
        num_passes: int = 3,
        mask_rate: Optional[float] = None,
    ) -> List[ClozeMaskResult]:
        """Generate multiple randomized cloze masks for Monte Carlo confidence."""
        passes = max(1, min(10, num_passes))
        results = []
        for i in range(passes):
            res = self.mask_text(text, mask_rate=mask_rate, pass_index=i)
            results.append(res)
        return results
