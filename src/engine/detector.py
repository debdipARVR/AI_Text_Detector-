"""Pass 2 & Pass 3 Sentence Cloze Congruence AI Text Detector.

Executes structured sentence-level masking across multi-passage essays:
- Pass 2 (Alternate): Removes alternate sentences every 2 lines.
- Pass 3 (Middle 3-Sentence): Removes 3 sentences from the middle of each passage,
  preserving the opening topic sentence and closing sentence as context anchors.

Evaluates propositional meaning, semantic cosine, phrasing alignment, and lexical overlap:
  Congruence = (0.40 * Meaning) + (0.40 * Cosine) + (0.10 * Semantic) + (0.10 * Lexical)
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from .cloze_masker import ClozeMasker, ClozeMaskResult, MaskedSpan, SentenceMetadata
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
    """Pass 2 & Pass 3 Sentence AI Text Detector powered by DeepEval Meaning Metrics and NVIDIA NIM."""

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
        """Perform Pass 2 & Pass 3 Sentence Cloze AI detection on input text."""
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
        # PASS 2: Alternate Sentence Masking (Sentences every 2 lines)
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

            infill_words = len(pred.split())
            target_words = span.metadata.word_count if span.metadata else len(span.original_text.split())
            span.infilled_word_count = infill_words
            span.word_count_delta = infill_words - target_words

            cos = compute_cosine_similarity(span.original_text, pred)
            sem = compute_semantic_congruence(span.original_text, pred)
            lex = compute_lexical_similarity(span.original_text, pred)
            meaning = compute_meaning_similarity(span.original_text, pred)
            
            # Dynamic Adaptive Scoring:
            # Baseline: 80% Meaning + 20% Cosine
            # When Cosine >= 70% (0.70): shifts to 60% Meaning + 40% Cosine
            if cos >= 0.70:
                comp = (0.60 * meaning) + (0.40 * cos)
            else:
                comp = (0.80 * meaning) + (0.20 * cos)

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
        # PASS 3: Three Middle Sentences Removed per Passage
        # =========================================================================
        pass3_mask = self.masker.mask_pass_3(cleaned_text)
        pass3_preds = self.nim_client.infill_cloze_spans(
            masked_text=pass3_mask.masked_text,
            spans=pass3_mask.spans,
            model_name=model,
            temperature=temperature,
        )

        pass3_orig_sentences: List[str] = []
        pass3_pred_sentences: List[str] = []
        pass3_reconstructed = pass3_mask.masked_text

        for span in pass3_mask.spans:
            pred = pass3_preds.get(span.placeholder, span.original_text)
            span.predicted_text = pred
            pass3_reconstructed = pass3_reconstructed.replace(span.placeholder, pred)

            pass3_orig_sentences.append(span.original_text)
            pass3_pred_sentences.append(pred)

            infill_words = len(pred.split())
            target_words = span.metadata.word_count if span.metadata else len(span.original_text.split())
            span.infilled_word_count = infill_words
            span.word_count_delta = infill_words - target_words

            cos = compute_cosine_similarity(span.original_text, pred)
            sem = compute_semantic_congruence(span.original_text, pred)
            lex = compute_lexical_similarity(span.original_text, pred)
            meaning = compute_meaning_similarity(span.original_text, pred)
            
            # Dynamic Adaptive Scoring:
            # Baseline: 80% Meaning + 20% Cosine
            # When Cosine >= 70% (0.70): shifts to 60% Meaning + 40% Cosine
            if cos >= 0.70:
                comp = (0.60 * meaning) + (0.40 * cos)
            else:
                comp = (0.80 * meaning) + (0.20 * cos)

            span.cosine_similarity = round(cos * 100.0, 1)
            span.semantic_similarity = round(sem * 100.0, 1)
            span.lexical_similarity = round(lex * 100.0, 1)
            span.meaning_similarity = round(meaning * 100.0, 1)
            span.composite_congruence = round(comp * 100.0, 1)
            span.status = classify_span_congruence(comp)

        # DeepEval Meaning & Congruence Evaluation for Pass 3
        pass3_eval = self.deepeval_evaluator.evaluate_sentence_pairs(
            masked_context=pass3_mask.masked_text,
            infilled_sentences=pass3_pred_sentences,
            original_sentences=pass3_orig_sentences,
        )

        # =========================================================================
        # PASS 2 & PASS 3 SYNTHESIS & FINAL VERDICT
        # =========================================================================
        p2_score = pass2_eval["congruence_score_percent"]
        p3_score = pass3_eval["congruence_score_percent"]
        verdict_data = compute_two_pass_verdict(p2_score, p3_score)

        avg_meaning = round((0.50 * pass2_eval.get("meaning_similarity_percent", 0.0)) + (0.50 * pass3_eval.get("meaning_similarity_percent", 0.0)), 1)
        avg_cosine = round((0.50 * pass2_eval.get("semantic_cosine_percent", 0.0)) + (0.50 * pass3_eval.get("semantic_cosine_percent", 0.0)), 1)
        avg_sem = round((0.50 * pass2_eval.get("semantic_similarity_percent", 0.0)) + (0.50 * pass3_eval.get("semantic_similarity_percent", 0.0)), 1)
        avg_lexical = round((0.50 * pass2_eval.get("lexical_similarity_percent", 0.0)) + (0.50 * pass3_eval.get("lexical_similarity_percent", 0.0)), 1)

        highlighted_html = self._render_highlighted_html(cleaned_text, pass3_mask.spans)

        return {
            "ai_probability": verdict_data["ai_probability"],
            "verdict": verdict_data["verdict"],
            "confidence": verdict_data["confidence"],
            "combined_congruence_score": verdict_data["combined_congruence_score"],
            "two_pass_verdict_reason": verdict_data["reason"],
            "pass_2": {
                "name": "Pass 2 (Alternate: sentences removed every 2 lines)",
                "sentences_masked_count": pass2_mask.masked_sentences_count,
                "masked_text": pass2_mask.masked_text,
                "reconstructed_text": pass2_reconstructed,
                "meaning_similarity": pass2_eval["meaning_similarity_percent"],
                "semantic_cosine": pass2_eval["semantic_cosine_percent"],
                "semantic_similarity": pass2_eval["semantic_similarity_percent"],
                "lexical_similarity": pass2_eval["lexical_similarity_percent"],
                "congruence_score": pass2_eval["congruence_score_percent"],
                "deepeval_reason": pass2_eval["reason"],
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original_sentence": s.original_text,
                        "predicted_sentence": s.predicted_text,
                        "paragraph_idx": s.paragraph_idx,
                        "meaning_similarity": s.meaning_similarity,
                        "semantic_cosine": s.cosine_similarity,
                        "semantic_similarity": s.semantic_similarity,
                        "lexical_similarity": s.lexical_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in pass2_mask.spans
                ],
            },
            "pass_3": {
                "name": "Pass 3 (Middle: 3 sentences removed per passage)",
                "sentences_masked_count": pass3_mask.masked_sentences_count,
                "masked_text": pass3_mask.masked_text,
                "reconstructed_text": pass3_reconstructed,
                "meaning_similarity": pass3_eval["meaning_similarity_percent"],
                "semantic_cosine": pass3_eval["semantic_cosine_percent"],
                "semantic_similarity": pass3_eval["semantic_similarity_percent"],
                "lexical_similarity": pass3_eval["lexical_similarity_percent"],
                "congruence_score": pass3_eval["congruence_score_percent"],
                "deepeval_reason": pass3_eval["reason"],
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original_sentence": s.original_text,
                        "predicted_sentence": s.predicted_text,
                        "paragraph_idx": s.paragraph_idx,
                        "meaning_similarity": s.meaning_similarity,
                        "semantic_cosine": s.cosine_similarity,
                        "semantic_similarity": s.semantic_similarity,
                        "lexical_similarity": s.lexical_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in pass3_mask.spans
                ],
            },
            # Compatibility alias
            "pass_1": {
                "name": "Pass 2 (Alternate: sentences removed every 2 lines)",
                "sentences_masked_count": pass2_mask.masked_sentences_count,
                "congruence_score": pass2_eval["congruence_score_percent"],
                "meaning_similarity": pass2_eval["meaning_similarity_percent"],
                "semantic_cosine": pass2_eval["semantic_cosine_percent"],
                "semantic_similarity": pass2_eval["semantic_similarity_percent"],
                "lexical_similarity": pass2_eval["lexical_similarity_percent"],
                "deepeval_reason": pass2_eval["reason"],
                "spans": [
                    {
                        "placeholder": s.placeholder,
                        "original_sentence": s.original_text,
                        "predicted_sentence": s.predicted_text,
                        "meaning_similarity": s.meaning_similarity,
                        "semantic_cosine": s.cosine_similarity,
                        "congruence": s.composite_congruence,
                        "status": s.status,
                    }
                    for s in pass2_mask.spans
                ],
            },
            "metrics": {
                "meaning_similarity_avg": avg_meaning,
                "semantic_cosine_avg": avg_cosine,
                "semantic_similarity_avg": avg_sem,
                "word_similarity_avg": avg_lexical,
                "congruence_avg": verdict_data["combined_congruence_score"],
                "weights": {
                    "policy": "Dynamic Adaptive Scoring (70% Cosine Threshold)",
                    "baseline_weights": "80% DeepEval Meaning + 20% Semantic Cosine (when Cosine < 70%)",
                    "boosted_weights": "60% DeepEval Meaning + 40% Semantic Cosine (when Cosine >= 70%)",
                },
                "burstiness": burstiness_info,
            },
            "deepeval_evaluation": {
                "framework": "DeepEval Meaning & Congruence Framework (Pass 2 & Pass 3)",
                "meaning_similarity_score": avg_meaning,
                "semantic_cosine_score": avg_cosine,
                "combined_congruence": verdict_data["combined_congruence_score"],
                "geval_score": verdict_data["combined_congruence_score"],
                "reason": f"Pass 2: {pass2_eval['reason']} | Pass 3: {pass3_eval['reason']}",
                "evaluator_model": model,
                "pass2_reason": pass2_eval["reason"],
                "pass3_reason": pass3_eval["reason"],
            },
            "parameters": {
                "model_name": model,
                "client_mode": self.nim_client.get_status()["mode"],
                "is_live_api": self.nim_client.is_live,
                "evaluation_framework": "DeepEval Pass 2 & Pass 3 Middle-Sentence Cloze Framework",
            },
            "primary_masked_text": pass3_mask.masked_text,
            "reconstructed_text": pass3_reconstructed,
            "spans": [
                {
                    "id": s.mask_id,
                    "placeholder": s.placeholder,
                    "original": s.original_text,
                    "predicted": s.predicted_text,
                    "meaning_similarity": s.meaning_similarity,
                    "semantic_cosine": s.cosine_similarity,
                    "semantic_similarity": s.semantic_similarity,
                    "lexical_similarity": s.lexical_similarity,
                    "congruence": s.composite_congruence,
                    "status": s.status,
                }
                for s in pass3_mask.spans
            ],
            "highlighted_html": highlighted_html,
        }

    def _render_highlighted_html(self, original_text: str, spans: List[MaskedSpan]) -> str:
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
                f"Key: {s.placeholder}&#10;"
                f"Original: {escaped_orig}&#10;"
                f"NIM Infill: {escaped_pred}&#10;"
                f"Meaning (40%): {s.meaning_similarity}% | Cosine (40%): {s.cosine_similarity}%&#10;"
                f"Congruence: {s.composite_congruence}% ({s.status})"
            )
            
            replacement = (
                f'<span class="cloze-span {badge_class}" title="{tooltip}" data-mask-id="{s.mask_id}">'
                f'<span class="span-text">{escaped_orig}</span>'
                f'<span class="span-badge">{s.placeholder} • Meaning: {s.meaning_similarity}%</span>'
                f'</span>'
            )
            
            if escaped_orig in result_html:
                result_html = result_html.replace(escaped_orig, replacement, 1)

        return result_html
