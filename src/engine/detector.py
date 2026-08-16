"""Cloze Congruence AI Text Detector.

The master detection engine combining randomized span masking, NVIDIA NIM
cloze infilling, multi-dimensional similarity scoring, and statistical attribution.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from .cloze_masker import ClozeMasker, MaskedSpan
from .metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_ai_probability,
    compute_lexical_similarity,
    compute_semantic_congruence,
)
from .nim_client import NvidiaNIMClient


class ClozeCongruenceDetector:
    """Detects AI-generated text by measuring cloze infill congruence with high-capacity LLMs."""

    def __init__(
        self,
        nim_client: Optional[NvidiaNIMClient] = None,
        default_mask_rate: float = 0.30,
        default_passes: int = 2,
    ):
        self.nim_client = nim_client or NvidiaNIMClient()
        self.masker = ClozeMasker(default_mask_rate=default_mask_rate)
        self.default_passes = default_passes

    def analyze(
        self,
        text: str,
        mask_rate: Optional[float] = None,
        num_passes: Optional[int] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Perform end-to-end Cloze Congruence AI detection on the input text.
        
        Args:
            text: Input text to analyze
            mask_rate: Fraction of words to mask (0.10 to 0.50, default 0.30)
            num_passes: Number of randomized Monte Carlo masking passes (1 to 5)
            model_name: NVIDIA NIM model name
            temperature: LLM temperature (0.0 for deterministic infill)
            
        Returns:
            Dictionary containing AI probability, metrics, span breakdowns, and visual HTML
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return {
                "ai_probability": 0.0,
                "verdict": "Empty Text",
                "confidence": "None",
                "error": "Input text is empty.",
            }

        passes_count = num_passes or self.default_passes
        passes_count = max(1, min(5, passes_count))
        
        rate = mask_rate if mask_rate is not None else self.masker.default_mask_rate
        rate = max(0.10, min(0.50, rate))

        # Calculate stylometric burstiness
        burstiness_info = calculate_burstiness(cleaned_text)

        all_span_congruences: List[float] = []
        all_lexical_sims: List[float] = []
        all_semantic_sims: List[float] = []
        primary_spans: List[MaskedSpan] = []
        primary_masked_text: str = ""

        pass_runs: List[Dict[str, Any]] = []

        for p_idx in range(passes_count):
            mask_result = self.masker.mask_text(
                cleaned_text,
                mask_rate=rate,
                pass_index=p_idx,
            )

            # Infill with NVIDIA NIM
            predictions = self.nim_client.infill_cloze_spans(
                masked_text=mask_result.masked_text,
                spans=mask_result.spans,
                model_name=model_name,
                temperature=temperature,
            )

            # Score each span in this pass
            for span in mask_result.spans:
                predicted = predictions.get(span.placeholder, span.original_text)
                span.predicted_text = predicted

                lex_sim = compute_lexical_similarity(span.original_text, predicted)
                sem_sim = compute_semantic_congruence(span.original_text, predicted)
                composite = (0.55 * sem_sim) + (0.45 * lex_sim)

                span.lexical_similarity = round(lex_sim * 100.0, 1)
                span.semantic_similarity = round(sem_sim * 100.0, 1)
                span.composite_congruence = round(composite * 100.0, 1)
                span.status = classify_span_congruence(composite)

                all_span_congruences.append(composite)
                all_lexical_sims.append(lex_sim)
                all_semantic_sims.append(sem_sim)

            if p_idx == 0:
                primary_spans = mask_result.spans
                primary_masked_text = mask_result.masked_text

            pass_runs.append({
                "pass_index": p_idx + 1,
                "masked_text": mask_result.masked_text,
                "spans_count": len(mask_result.spans),
                "masked_words": mask_result.masked_words,
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original": s.original_text,
                        "predicted": s.predicted_text,
                        "lexical_similarity": s.lexical_similarity,
                        "semantic_similarity": s.semantic_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in mask_result.spans
                ],
            })

        # Calculate final AI Probability
        ai_prob_data = compute_ai_probability(
            span_congruences=all_span_congruences,
            lexical_similarities=all_lexical_sims,
            semantic_similarities=all_semantic_sims,
            burstiness_score=burstiness_info["burstiness_score"],
        )

        # Generate interactive HTML snippet with visual highlights
        highlighted_html = self._render_highlighted_html(cleaned_text, primary_spans)

        return {
            "ai_probability": ai_prob_data["ai_probability"],
            "verdict": ai_prob_data["verdict"],
            "confidence": ai_prob_data["confidence"],
            "metrics": {
                "semantic_similarity_avg": ai_prob_data["semantic_similarity_avg"],
                "word_similarity_avg": ai_prob_data["lexical_similarity_avg"],
                "congruence_avg": ai_prob_data["congruence_avg"],
                "congruent_ratio": ai_prob_data["congruent_ratio"],
                "congruent_spans_count": ai_prob_data["congruent_spans_count"],
                "total_spans_count": ai_prob_data["total_spans_count"],
                "burstiness": burstiness_info,
            },
            "parameters": {
                "mask_rate": rate,
                "num_passes": passes_count,
                "model_name": model_name or self.nim_client.default_model,
                "client_mode": self.nim_client.get_status()["mode"],
                "is_live_api": self.nim_client.is_live,
            },
            "primary_masked_text": primary_masked_text,
            "spans": [
                {
                    "id": s.mask_id,
                    "placeholder": s.placeholder,
                    "original": s.original_text,
                    "predicted": s.predicted_text,
                    "lexical_similarity": s.lexical_similarity,
                    "semantic_similarity": s.semantic_similarity,
                    "congruence": s.composite_congruence,
                    "status": s.status,
                }
                for s in primary_spans
            ],
            "passes_details": pass_runs,
            "highlighted_html": highlighted_html,
        }

    def _render_highlighted_html(self, original_text: str, spans: List[MaskedSpan]) -> str:
        """Render annotated HTML with colored inline tags for each evaluated span."""
        if not spans:
            return html.escape(original_text)

        result_html = html.escape(original_text)
        # Substitute in reverse order or by string match to avoid offset collisions
        for s in spans:
            escaped_orig = html.escape(s.original_text)
            escaped_pred = html.escape(s.predicted_text or "")
            
            badge_class = "span-congruent" if s.status == "CONGRUENT" else (
                "span-partial" if s.status == "PARTIAL" else "span-divergent"
            )
            
            tooltip = f"Original: &quot;{escaped_orig}&quot; &#10;AI Infill: &quot;{escaped_pred}&quot; &#10;Congruence: {s.composite_congruence}% ({s.status})"
            
            replacement = (
                f'<span class="cloze-span {badge_class}" title="{tooltip}" data-mask-id="{s.mask_id}">'
                f'<span class="span-text">{escaped_orig}</span>'
                f'<span class="span-badge">{s.composite_congruence}%</span>'
                f'</span>'
            )
            
            if escaped_orig in result_html:
                result_html = result_html.replace(escaped_orig, replacement, 1)

        return result_html
