"""Pass 2 & Pass 3 Sentence-Level Cloze Masking Engine for AI Text Detection.

Designed for multi-passage essays (7-10+ passages):
- Pass 2 (Alternate): Removes alternate sentences across every 2 lines.
- Pass 3 (Middle 3-Sentence Removal): Removes 3 sentences from the middle of each passage/paragraph,
  preserving the opening topic sentence and closing sentence as context anchors.
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
        spec_chars_str = ", ".join(f"'{c}'" for c in sorted(set(self.special_characters))) if self.special_characters else "None"
        punct_desc = ", ".join(f"{k}: {v}" for k, v in self.punctuation_counts.items() if v > 0)
        return (
            f"Word Count: ~{self.word_count} | "
            f"Spaces: {self.space_count} | "
            f"Special Chars: [{spec_chars_str}] | "
            f"Punctuation: ({punct_desc or 'Standard period'})"
        )


@dataclass
class MaskedSpan:
    """Represents a masked complete sentence span with structural metadata."""
    mask_id: int
    placeholder: str
    original_text: str
    sentence_idx: int
    paragraph_idx: int = 0
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
    total_paragraphs: int = 0
    total_words: int = 0
    masked_words: int = 0
    mask_ratio: float = 0.0
    pass_index: int = 2
    pass_name: str = "Pass 2 (Alternate: sentences removed every 2 lines)"


class ClozeMasker:
    """Masks complete sentences into [1], [2], [3] placeholder keys for Pass 2 and Pass 3."""

    def __init__(self, default_mask_rate: float = 0.30):
        self.default_mask_rate = default_mask_rate

    @staticmethod
    def extract_sentence_metadata(sentence: str) -> SentenceMetadata:
        words = re.findall(r'\b[a-zA-Z0-9_\'-]+\b', sentence)
        word_count = len(words)
        char_count = len(sentence)
        space_count = sentence.count(" ") + sentence.count("\t")

        special_chars = [c for c in sentence if not c.isalnum() and not c.isspace()]
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
    def split_into_paragraphs(text: str) -> List[List[str]]:
        """Split multi-passage essay into paragraphs, and each paragraph into clean sentences."""
        cleaned = text.strip()
        if not cleaned:
            return []

        raw_paragraphs = re.split(r'\n{2,}|\r\n\r\n', cleaned)
        paragraphs: List[List[str]] = []

        for p in raw_paragraphs:
            p = p.strip()
            if not p:
                continue
            
            # Skip standalone markdown headers (e.g. '# Title')
            if p.startswith("#") and len(p.split()) < 8:
                continue
            if re.match(r'^(?:[IVXLCDM]+\.?|[0-9]+\.?)$', p, re.IGNORECASE):
                continue

            sentences: List[str] = []
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', p)
            for part in parts:
                s = part.strip()
                if len(s.split()) >= 1 and not s.startswith("#") and not re.match(r'^(?:[IVXLCDM]+\.?|[0-9]+\.?)$', s, re.IGNORECASE):
                    sentences.append(s)

            if sentences:
                paragraphs.append(sentences)

        return paragraphs

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Flattened list of sentences across all paragraphs."""
        paragraphs = cls.split_into_paragraphs(text)
        flat: List[str] = []
        for p in paragraphs:
            flat.extend(p)
        return flat

    def mask_pass_2(self, text: str) -> ClozeMaskResult:
        """Pass 2: Alternate sentence removal (every 2 lines across paragraphs)."""
        paragraphs = self.split_into_paragraphs(text)
        if not paragraphs:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=2)

        spans: List[MaskedSpan] = []
        masked_paragraph_blocks: List[str] = []
        mask_counter = 1
        global_s_idx = 0

        total_sentences = sum(len(p) for p in paragraphs)
        total_words = sum(sum(len(s.split()) for s in p) for p in paragraphs)
        masked_words = 0

        for p_idx, p_sentences in enumerate(paragraphs):
            n = len(p_sentences)
            masked_p = list(p_sentences)

            # Alternate masking: 1, 3, 5... (leaves 0 as opening topic sentence)
            for i in range(1, n, 2):
                orig_s = p_sentences[i]
                placeholder = f"[{mask_counter}]"
                masked_p[i] = placeholder
                
                meta = self.extract_sentence_metadata(orig_s)
                masked_words += meta.word_count

                prefix = p_sentences[i - 1] if i > 0 else ""
                suffix = p_sentences[i + 1] if i + 1 < n else ""

                spans.append(
                    MaskedSpan(
                        mask_id=mask_counter,
                        placeholder=placeholder,
                        original_text=orig_s,
                        sentence_idx=global_s_idx + i,
                        paragraph_idx=p_idx,
                        metadata=meta,
                        context_prefix=prefix,
                        context_suffix=suffix,
                    )
                )
                mask_counter += 1

            global_s_idx += n
            masked_paragraph_blocks.append(" ".join(masked_p))

        masked_text = "\n\n".join(masked_paragraph_blocks)
        ratio = round((masked_words / max(1, total_words)), 3)

        return ClozeMaskResult(
            original_text=text,
            masked_text=masked_text,
            spans=spans,
            total_sentences=total_sentences,
            masked_sentences_count=len(spans),
            total_paragraphs=len(paragraphs),
            total_words=total_words,
            masked_words=masked_words,
            mask_ratio=ratio,
            pass_index=2,
            pass_name="Pass 2 (Alternate: sentences removed every 2 lines)",
        )

    def mask_pass_3(self, text: str) -> ClozeMaskResult:
        """Pass 3: Three sentences removed from the middle of each passage/paragraph.
        
        Preserves the opening topic sentence (S_0) and concluding sentence (S_-1) as anchors.
        """
        paragraphs = self.split_into_paragraphs(text)
        if not paragraphs:
            return ClozeMaskResult(original_text=text, masked_text="", total_sentences=0, pass_index=3)

        spans: List[MaskedSpan] = []
        masked_paragraph_blocks: List[str] = []
        mask_counter = 1
        global_s_idx = 0

        total_sentences = sum(len(p) for p in paragraphs)
        total_words = sum(sum(len(s.split()) for s in p) for p in paragraphs)
        masked_words = 0

        for p_idx, p_sentences in enumerate(paragraphs):
            n = len(p_sentences)
            masked_p = list(p_sentences)

            # Determine 3 middle indices for this passage
            target_indices: List[int] = []
            if n >= 5:
                # E.g. for n=5 (0,1,2,3,4) -> remove 1,2,3 (keeping 0 and 4)
                # E.g. for n=6 (0,1,2,3,4,5) -> remove 2,3,4 (keeping 0,1 and 5)
                mid = n // 2
                start_idx = max(1, mid - 1)
                end_idx = min(n - 1, start_idx + 3)
                target_indices = list(range(start_idx, end_idx))
                # Ensure exactly 3 if possible
                if len(target_indices) < 3 and n >= 4:
                    target_indices = [1, 2, 3] if n >= 4 else [1, 2]
            elif n == 4:
                # 4 sentences: keep S_0, remove middle 3: [1, 2, 3]
                target_indices = [1, 2, 3]
            elif n == 3:
                # 3 sentences: keep S_0 and S_2, remove middle 1: [1]
                target_indices = [1]
            elif n == 2:
                target_indices = [1]
            else:
                target_indices = [0]

            for i in target_indices:
                if i >= n:
                    continue
                orig_s = p_sentences[i]
                placeholder = f"[{mask_counter}]"
                masked_p[i] = placeholder
                
                meta = self.extract_sentence_metadata(orig_s)
                masked_words += meta.word_count

                prefix = p_sentences[i - 1] if i > 0 else ""
                suffix = p_sentences[i + 1] if i + 1 < n else ""

                spans.append(
                    MaskedSpan(
                        mask_id=mask_counter,
                        placeholder=placeholder,
                        original_text=orig_s,
                        sentence_idx=global_s_idx + i,
                        paragraph_idx=p_idx,
                        metadata=meta,
                        context_prefix=prefix,
                        context_suffix=suffix,
                    )
                )
                mask_counter += 1

            global_s_idx += n
            masked_paragraph_blocks.append(" ".join(masked_p))

        masked_text = "\n\n".join(masked_paragraph_blocks)
        ratio = round((masked_words / max(1, total_words)), 3)

        return ClozeMaskResult(
            original_text=text,
            masked_text=masked_text,
            spans=spans,
            total_sentences=total_sentences,
            masked_sentences_count=len(spans),
            total_paragraphs=len(paragraphs),
            total_words=total_words,
            masked_words=masked_words,
            mask_ratio=ratio,
            pass_index=3,
            pass_name="Pass 3 (Middle: 3 sentences removed from passage center)",
        )

    def mask_text(
        self,
        text: str,
        mask_rate: Optional[float] = None,
        pass_index: int = 2,
    ) -> ClozeMaskResult:
        if pass_index == 3:
            return self.mask_pass_3(text)
        return self.mask_pass_2(text)
