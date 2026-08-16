"""Unit and integration tests for DeepEval framework metrics, Key-Value pairing, and custom evaluator."""

from src.deepeval_model_connect import NvidiaLLM_Understanding, DeepEvalCongruencyEvaluator
from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient
from src.engine.cloze_masker import MaskedSpan


def test_deepeval_evaluator_creation():
    evaluator_glm = NvidiaLLM_Understanding(model_name="z-ai/glm-5.2", api_key="")
    assert evaluator_glm.get_model_name() == "z-ai/glm-5.2"
    
    evaluator_inkling = NvidiaLLM_Understanding(model_name="thinkingmachines/inkling", api_key="")
    assert evaluator_inkling.get_model_name() == "thinkingmachines/inkling"
    
    test_gen = evaluator_glm.generate("Test prompt furthermore crucial")
    assert isinstance(test_gen, str)
    assert len(test_gen) > 0


def test_key_value_paired_sentence_evaluation():
    congruence_eval = DeepEvalCongruencyEvaluator(model_name="z-ai/glm-5.2", api_key="")
    
    # Original sentences
    orig_s1 = "Artificial intelligence has revolutionized modern technological paradigms."
    orig_s2 = "Foundational models synthesize highly structured responses from large-scale pretraining datasets."
    
    # Newly infilled sentences (key-value paired)
    infilled_s1 = "Artificial intelligence has revolutionized modern technological paradigms."
    infilled_s2 = "Foundational models synthesize structured responses across diverse computational domains."

    masked_context = "[MASK_1] Furthermore, deep learning models learn representations. [MASK_2]"

    res = congruence_eval.evaluate_sentence_pairs(
        masked_context=masked_context,
        infilled_sentences=[infilled_s1, infilled_s2],
        original_sentences=[orig_s1, orig_s2],
    )

    assert "pair_evaluations" in res
    assert len(res["pair_evaluations"]) == 2
    
    # Verify Pair 1 exact matching
    pair1 = res["pair_evaluations"][0]
    assert pair1["original_sentence"] == orig_s1
    assert pair1["predicted_sentence"] == infilled_s1
    assert pair1["meaning_similarity"] >= 85.0

    # Verify Pair 2 exact matching
    pair2 = res["pair_evaluations"][1]
    assert pair2["original_sentence"] == orig_s2
    assert pair2["predicted_sentence"] == infilled_s2
    assert pair2["meaning_similarity"] >= 50.0


def test_detector_with_deepeval_framework():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    text = (
        "Furthermore, artificial intelligence plays a crucial role in modern computational paradigms. "
        "Moreover, navigating the multifaceted landscape of deep learning fosters transformative breakthroughs."
    )
    res = detector.analyze(text, mask_rate=0.30, num_passes=2, model_name="z-ai/glm-5.2")
    
    assert "deepeval_evaluation" in res
    assert "meaning_similarity_score" in res["deepeval_evaluation"]
    assert "combined_congruence" in res["deepeval_evaluation"]
    assert "pass_1" in res
    assert "pass_2" in res
    assert "DeepEval" in res["deepeval_evaluation"]["framework"]
    assert res["parameters"]["model_name"] == "z-ai/glm-5.2"
