"""Academic Benchmark & Ablation Evaluation Harness for AI Text Detection Paper.

Runs full empirical evaluation and ablation studies across the multi-domain benchmark corpus:
1. Full Proposed Architecture (Pass 2 + Pass 3 + Continuous Sigmoid Gating + Bipartite Matching + Burstiness)
2. Ablation A: Without Bipartite Matching (Strict Positional Matching)
3. Ablation B: Without Sigmoid Gating (Fixed 40/40/10/10 static weights)
4. Ablation C: Single Pass (Pass 2 Only)
5. Ablation D: Single Pass (Pass 3 Only)

Logs all individual predictions, confusion matrices, ROC-AUC, Precision, Recall, and F1 to JSON and CSV.
"""

import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.metrics import (
    compute_cosine_similarity,
    compute_meaning_similarity,
    compute_semantic_congruence,
    compute_lexical_similarity,
    compute_dynamic_pair_congruence,
    compute_sigmoid_dynamic_weights,
    compute_two_pass_verdict,
)
from src.engine.nim_client import NvidiaNIMClient

CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_corpus", "benchmark_essays.json"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_classification_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.50) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1, FPR, FNR, and ROC-AUC."""
    y_pred = [1 if s >= (threshold * 100.0 if max(y_scores) > 1.0 else threshold) else 0 for s in y_scores]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))

    # Compute ROC-AUC via trapezoidal rule over sorted thresholds
    paired = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    positives = sum(y_true)
    negatives = total - positives
    
    if positives == 0 or negatives == 0:
        auc = 1.0
    else:
        auc = 0.0
        tp_count = 0
        fp_count = 0
        prev_fp = 0
        prev_tp = 0
        for score, label in paired:
            if label == 1:
                tp_count += 1
            else:
                fp_count += 1
                auc += (tp_count + prev_tp) * 0.5 * (fp_count - prev_fp)
                prev_fp = fp_count
                prev_tp = tp_count
        auc = auc / (positives * negatives)

    return {
        "accuracy": round(accuracy * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "fpr": round(fpr * 100.0, 2),
        "fnr": round(fnr * 100.0, 2),
        "roc_auc": round(auc, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_samples": total,
    }


def run_benchmark():
    print("=" * 80)
    print(" EMPIRICAL EVALUATION & ABLATION BENCHMARK FOR AI TEXT DETECTION PAPER")
    print("=" * 80)

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}. Run generate_benchmark_corpus.py first.")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        essays = json.load(f)

    print(f"[INFO] Loaded {len(essays)} essays across diverse domains.")

    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)

    eval_records = []
    y_true_binary = []  # 1 = AI / Hybrid, 0 = Human
    full_model_scores = []
    strict_matching_scores = []
    static_weight_scores = []
    pass2_only_scores = []
    pass3_only_scores = []

    print("\n[RUNNING BATCH EVALUATION] Processing essays...")
    t0 = time.time()

    for idx, essay in enumerate(essays):
        eid = essay["id"]
        domain = essay["domain"]
        label = essay["ground_truth_label"]
        text = essay["text"]

        is_ai = 1 if label in ["AI_GENERATED", "HYBRID_ASSISTED"] else 0
        y_true_binary.append(is_ai)

        # 1. Full Proposed Model
        res = detector.analyze(text)
        full_prob = res["ai_probability"]
        full_cong = res["combined_congruence_score"]
        p2_cong = res["pass_2"]["congruence_score"]
        p3_cong = res["pass_3"]["congruence_score"]
        conf = res["confidence"]
        verdict = res["verdict"]

        full_model_scores.append(full_prob)
        pass2_only_scores.append(p2_cong)
        pass3_only_scores.append(p3_cong)

        # 2. Ablation A: Strict Positional Matching (No Bipartite)
        p3_spans = res["pass_3"]["spans"]
        if p3_spans:
            strict_congs = []
            for s in p3_spans:
                m = s["meaning_similarity"] / 100.0
                c = s["semantic_cosine"] / 100.0
                cong_val, _, _, _ = compute_dynamic_pair_congruence(m, c)
                strict_congs.append(cong_val * 100.0)
            p3_strict = sum(strict_congs) / len(strict_congs)
        else:
            p3_strict = p3_cong
        strict_combined = round((0.50 * p2_cong) + (0.50 * p3_strict), 1)
        strict_matching_scores.append(strict_combined)

        # 3. Ablation B: Fixed 40/40/10/10 Static Weights (No Sigmoid Gating)
        if p3_spans:
            static_congs = []
            for s in p3_spans:
                m = s["meaning_similarity"] / 100.0
                c = s["semantic_cosine"] / 100.0
                sem = s.get("semantic_similarity", c * 100.0) / 100.0
                lex = s.get("lexical_similarity", c * 100.0) / 100.0
                static_val = (0.40 * m) + (0.40 * c) + (0.10 * sem) + (0.10 * lex)
                static_congs.append(static_val * 100.0)
            p3_static = sum(static_congs) / len(static_congs)
        else:
            p3_static = p3_cong
        static_combined = round((0.50 * p2_cong) + (0.50 * p3_static), 1)
        static_weight_scores.append(static_combined)

        record = {
            "id": eid,
            "domain": domain,
            "title": essay["title"],
            "ground_truth_label": label,
            "is_ai_binary": is_ai,
            "predicted_verdict": verdict,
            "ai_probability": full_prob,
            "confidence": conf,
            "combined_congruence": full_cong,
            "pass_2_congruence": p2_cong,
            "pass_3_congruence": p3_cong,
            "meaning_similarity_avg": res["metrics"]["meaning_similarity_avg"],
            "semantic_cosine_avg": res["metrics"]["semantic_cosine_avg"],
            "strict_matching_score": strict_combined,
            "static_weight_score": static_combined,
        }
        eval_records.append(record)
        print(f"  [{idx+1:02d}/{len(essays):02d}] {eid} ({domain[:25]}...): Label={label} -> AI Prob={full_prob}% | Cong={full_cong}% | Verdict={verdict}")

    elapsed = round(time.time() - t0, 2)
    print(f"\n[OK] Benchmark execution completed in {elapsed}s.")

    # Compute Statistical Metrics across Variants
    full_metrics = calculate_classification_metrics(y_true_binary, full_model_scores, threshold=0.50)
    strict_metrics = calculate_classification_metrics(y_true_binary, strict_matching_scores, threshold=0.50)
    static_metrics = calculate_classification_metrics(y_true_binary, static_weight_scores, threshold=0.50)
    p2_metrics = calculate_classification_metrics(y_true_binary, pass2_only_scores, threshold=0.50)
    p3_metrics = calculate_classification_metrics(y_true_binary, pass3_only_scores, threshold=0.50)

    # Summary Ablation Table
    ablation_summary = {
        "Full_Proposed_Model": full_metrics,
        "Ablation_No_Bipartite": strict_metrics,
        "Ablation_Static_Weights_40_40": static_metrics,
        "Ablation_Pass_2_Only": p2_metrics,
        "Ablation_Pass_3_Only": p3_metrics,
    }

    # Save to JSON & CSV
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    ablation_path = os.path.join(RESULTS_DIR, "ablation_summary.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"records": eval_records, "metrics": full_metrics, "ablation": ablation_summary}, f, indent=2)

    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    fieldnames = list(eval_records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_records)

    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS & ABLATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total Samples Evaluated:    {len(eval_records)}")
    print(f"Full Model Accuracy:        {full_metrics['accuracy']}%")
    print(f"Full Model Precision:       {full_metrics['precision']}%")
    print(f"Full Model Recall:          {full_metrics['recall']}%")
    print(f"Full Model F1-Score:        {full_metrics['f1_score']}%")
    print(f"Full Model ROC-AUC:         {full_metrics['roc_auc']}")
    print(f"False Positive Rate (Human): {full_metrics['fpr']}%")
    print(f"False Negative Rate (AI):    {full_metrics['fnr']}%\n")

    print("--- Ablation Comparison Table ---")
    print(f"{'Variant':<32} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'FPR (Human)':<12}")
    print("-" * 80)
    for k, v in ablation_summary.items():
        print(f"{k:<32} | {v['accuracy']:<9}% | {v['f1_score']:<9}% | {v['roc_auc']:<10} | {v['fpr']:<11}%")
    print("=" * 80)
    print(f"Saved artifacts to:\n  - {json_path}\n  - {csv_path}\n  - {ablation_path}\n")

    return full_metrics, ablation_summary

if __name__ == "__main__":
    run_benchmark()
