"""Two-Pass Complete Sentence Cloze Congruence AI Text Detector with DeepEval.

Executes two structured sentence-level masking passes:
- Pass 1 (Sparse): Removes 1 sentence every 4 lines.
- Pass 2 (Alternate): Removes alternate sentences every 2 lines.

Evaluates propositional Meaning Similarity (Highest Priority via DeepEval),
Semantic Cosine Similarity, and Lexical Overlap for both passes.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from .cloze_masker import ClozeMasker, ClozeMaskResult, MaskedSpan
from .metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_cosine_similarity,
    compute_lexical_similarity,
    compute_meaning_similarity,
    compute_semantic_congruence,
    compute_two_pass_verdict,
)
from .nim_client import NvidiaNIMClient
from ..deepeval_model_connect import DeepEvalCongruencyEvaluator


class ClozeCongruenceDetector:
    """Two-Pass Complete Sentence AI Text Detector powered by DeepEval Meaning Metrics and NVIDIA NIM."""

    def __init__(
        self,
        nim_client: Optional[NvidiaNIMClient] = None,
        default_mask_rate: float = 0.30,
        default_passes: int = 2,
    ):
        self.nim_client = nim_client or NvidiaNIMClient()
        self.masker = ClozeMasker(default_mask_rate=default_mask_rate)
        self.default_passes = default_passes
        self.deepeval_evaluator = DeepEvalCongruencyEvaluator(
            model_name=self.nim_client.default_model,
            api_key=self.nim_client.api_key if self.nim_client.is_live else "",
        )

    def analyze(
        self,
        text: str,
        mask_rate: Optional[float] = None,
        num_passes: Optional[int] = 2,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Perform Two-Pass Sentence Cloze AI detection on input text.
        
        Args:
            text: Input text/paragraph to analyze
            mask_rate: Unused (kept for compatibility)
            num_passes: Number of passes (1 or 2, default 2)
            model_name: NVIDIA NIM model (default: z-ai/glm-5.2)
            temperature: LLM temperature (0.0 for deterministic infill)
            
        Returns:
            Dictionary containing Pass 1, Pass 2, Meaning Similarity, Cosine Similarity,
            Congruency Scores, and Two-Pass Final Verdict.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return {
                "ai_probability": 0.0,
                "verdict": "Empty Text",
                "confidence": "None",
                "error": "Input text is empty.",
            }

        burstiness_info = calculate_burstiness(cleaned_text)
        model = model_name or self.nim_client.default_model

        # =========================================================================
        # PASS 1: Sparse Complete Sentence Masking (1 sentence per 4 lines)
        # =========================================================================
        pass1_mask = self.masker.mask_pass_1(cleaned_text)
        pass1_preds = self.nim_client.infill_cloze_spans(
            masked_text=pass1_mask.masked_text,
            spans=pass1_mask.spans,
            model_name=model,
            temperature=temperature,
        )

        pass1_orig_sentences: List[str] = []
        pass1_pred_sentences: List[str] = []
        pass1_reconstructed = pass1_mask.masked_text

        for span in pass1_mask.spans:
            pred = pass1_preds.get(span.placeholder, span.original_text)
            span.predicted_text = pred
            pass1_reconstructed = pass1_reconstructed.replace(span.placeholder, pred)

            pass1_orig_sentences.append(span.original_text)
            pass1_pred_sentences.append(pred)

            cos = compute_cosine_similarity(span.original_text, pred)
            sem = compute_semantic_congruence(span.original_text, pred)
            lex = compute_lexical_similarity(span.original_text, pred)
            meaning = compute_meaning_similarity(span.original_text, pred)
            comp = (0.55 * meaning) + (0.30 * cos) + (0.15 * lex)

            span.cosine_similarity = round(cos * 100.0, 1)
            span.semantic_similarity = round(sem * 100.0, 1)
            span.lexical_similarity = round(lex * 100.0, 1)
            span.meaning_similarity = round(meaning * 100.0, 1)
            span.composite_congruence = round(comp * 100.0, 1)
            span.status = classify_span_congruence(comp)

        # DeepEval Meaning & Congruence Evaluation for Pass 1
        pass1_eval = self.deepeval_evaluator.evaluate_sentence_pairs(
            masked_context=pass1_mask.masked_text,
            infilled_sentences=pass1_pred_sentences,
            original_sentences=pass1_orig_sentences,
        )

        # =========================================================================
        # PASS 2: Alternate Complete Sentence Masking (Sentences every 2 lines)
        # =========================================================================
        pass2_mask = self.masker.mask_pass_2(cleaned_text)
        pass2_preds = self.nim_client.infill_cloze_spans(
            masked_text=pass2_mask.masked_text,
            spans=pass2_mask.spans,
            model_name=model,
            temperature=temperature,
        )

        pass2_orig_sentences: List[str] = []
        pass2_pred_sentences: List[str] = []
        pass2_reconstructed = pass2_mask.masked_text

        for span in pass2_mask.spans:
            pred = pass2_preds.get(span.placeholder, span.original_text)
            span.predicted_text = pred
            pass2_reconstructed = pass2_reconstructed.replace(span.placeholder, pred)

            pass2_orig_sentences.append(span.original_text)
            pass2_pred_sentences.append(pred)

            cos = compute_cosine_similarity(span.original_text, pred)
            sem = compute_semantic_congruence(span.original_text, pred)
            lex = compute_lexical_similarity(span.original_text, pred)
            meaning = compute_meaning_similarity(span.original_text, pred)
            comp = (0.55 * meaning) + (0.30 * cos) + (0.15 * lex)

            span.cosine_similarity = round(cos * 100.0, 1)
            span.semantic_similarity = round(sem * 100.0, 1)
            span.lexical_similarity = round(lex * 100.0, 1)
            span.meaning_similarity = round(meaning * 100.0, 1)
            span.composite_congruence = round(comp * 100.0, 1)
            span.status = classify_span_congruence(comp)

        # DeepEval Meaning & Congruence Evaluation for Pass 2
        pass2_eval = self.deepeval_evaluator.evaluate_sentence_pairs(
            masked_context=pass2_mask.masked_text,
            infilled_sentences=pass2_pred_sentences,
            original_sentences=pass2_orig_sentences,
        )

        # =========================================================================
        # TWO-PASS SYNTHESIS & FINAL VERDICT
        # =========================================================================
        p1_score = pass1_eval["congruence_score_percent"]
        p2_score = pass2_eval["congruence_score_percent"]
        verdict_data = compute_two_pass_verdict(p1_score, p2_score)

        # Combined Meaning & Cosine Metrics
        avg_meaning = round((0.50 * pass1_eval["meaning_similarity_percent"]) + (0.50 * pass2_eval["meaning_similarity_percent"]), 1)
        avg_cosine = round((0.50 * pass1_eval["semantic_cosine_percent"]) + (0.50 * pass2_eval["semantic_cosine_percent"]), 1)
        avg_lexical = round((0.50 * pass1_eval["lexical_similarity_percent"]) + (0.50 * pass2_eval["lexical_similarity_percent"]), 1)

        # Render highlighted HTML snippet
        highlighted_html = self._render_highlighted_html(cleaned_text, pass2_mask.spans)

        return {
            "ai_probability": verdict_data["ai_probability"],
            "verdict": verdict_data["verdict"],
            "confidence": verdict_data["confidence"],
            "combined_congruence_score": verdict_data["combined_congruence_score"],
            "two_pass_verdict_reason": verdict_data["reason"],
            "pass_1": {
                "name": "Pass 1 (Sparse: 1 sentence removed per 4 lines)",
                "sentences_masked_count": pass1_mask.masked_sentences_count,
                "masked_text": pass1_mask.masked_text,
                "reconstructed_text": pass1_reconstructed,
                "meaning_similarity": pass1_eval["meaning_similarity_percent"],
                "semantic_cosine": pass1_eval["semantic_cosine_percent"],
                "lexical_similarity": pass1_eval["lexical_similarity_percent"],
                "congruence_score": pass1_eval["congruence_score_percent"],
                "deepeval_reason": pass1_eval["reason"],
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original_sentence": s.original_text,
                        "predicted_sentence": s.predicted_text,
                        "meaning_similarity": s.meaning_similarity,
                        "semantic_cosine": s.cosine_similarity,
                        "lexical_similarity": s.lexical_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in pass1_mask.spans
                ],
            },
            "pass_2": {
                "name": "Pass 2 (Alternate: sentences removed every 2 lines)",
                "sentences_masked_count": pass2_mask.masked_sentences_count,
                "masked_text": pass2_mask.masked_text,
                "reconstructed_text": pass2_reconstructed,
                "meaning_similarity": pass2_eval["meaning_similarity_percent"],
                "semantic_cosine": pass2_eval["semantic_cosine_percent"],
                "lexical_similarity": pass2_eval["lexical_similarity_percent"],
                "congruence_score": pass2_eval["congruence_score_percent"],
                "deepeval_reason": pass2_eval["reason"],
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original_sentence": s.original_text,
                        "predicted_sentence": s.predicted_text,
                        "meaning_similarity": s.meaning_similarity,
                        "semantic_cosine": s.cosine_similarity,
                        "lexical_similarity": s.lexical_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in pass2_mask.spans
                ],
            },
            "metrics": {
                "meaning_similarity_avg": avg_meaning,
                "semantic_cosine_avg": avg_cosine,
                "word_similarity_avg": avg_lexical,
                "congruence_avg": verdict_data["combined_congruence_score"],
                "weights": {
                    "meaning_similarity_weight": "55% (Highest Priority)",
                    "semantic_cosine_weight": "30%",
                    "lexical_overlap_weight": "15%",
                },
                "burstiness": burstiness_info,
            },
            "deepeval_evaluation": {
                "framework": "DeepEval Meaning & Congruence Framework",
                "meaning_similarity_score": avg_meaning,
                "semantic_cosine_score": avg_cosine,
                "combined_congruence": verdict_data["combined_congruence_score"],
                "geval_score": verdict_data["combined_congruence_score"],
                "reason": f"Pass 1: {pass1_eval['reason']} | Pass 2: {pass2_eval['reason']}",
                "evaluator_model": model,
                "pass1_reason": pass1_eval["reason"],
                "pass2_reason": pass2_eval["reason"],
            },
            "parameters": {
                "model_name": model,
                "client_mode": self.nim_client.get_status()["mode"],
                "is_live_api": self.nim_client.is_live,
                "evaluation_framework": "DeepEval Custom Meaning Metric + Cosine Similarity",
            },
            "primary_masked_text": pass2_mask.masked_text,
            "reconstructed_text": pass2_reconstructed,
            "spans": [
                {
                    "id": s.mask_id,
                    "placeholder": s.placeholder,
                    "original": s.original_text,
                    "predicted": s.predicted_text,
                    "meaning_similarity": s.meaning_similarity,
                    "semantic_similarity": s.semantic_similarity,
                    "lexical_similarity": s.lexical_similarity,
                    "congruence": s.composite_congruence,
                    "status": s.status,
                }
                for s in pass2_mask.spans
            ],
            "highlighted_html": highlighted_html,
        }

    def _render_highlighted_html(self, original_text: str, spans: List[MaskedSpan]) -> str:
        """Render annotated HTML with colored sentence tags."""
        if not spans:
            return html.escape(original_text)

        result_html = html.escape(original_text)
        for s in spans:
            escaped_orig = html.escape(s.original_text)
            escaped_pred = html.escape(s.predicted_text or "")
            
            badge_class = "span-congruent" if s.status == "CONGRUENT" else (
                "span-partial" if s.status == "PARTIAL" else "span-divergent"
            )
            
            tooltip = (
                f"Original: {escaped_orig}&#10;"
                f"NIM Infill: {escaped_pred}&#10;"
                f"Meaning: {s.meaning_similarity}% | Congruence: {s.composite_congruence}% ({s.status})"
            )
            
            replacement = (
                f'<span class="cloze-span {badge_class}" title="{tooltip}" data-mask-id="{s.mask_id}">'
                f'<span class="span-text">{escaped_orig}</span>'
                f'<span class="span-badge">Meaning: {s.meaning_similarity}%</span>'
                f'</span>'
            )
            
            if escaped_orig in result_html:
                result_html = result_html.replace(escaped_orig, replacement, 1)

        return result_html
