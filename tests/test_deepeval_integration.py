"""Unit and integration tests for DeepEval framework metrics and custom evaluator."""

from src.deepeval_model_connect import NvidiaLLM_Understanding, DeepEvalCongruencyEvaluator
from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient


def test_deepeval_evaluator_creation():
    # Test with offline evaluator for instant deterministic execution
    evaluator = NvidiaLLM_Understanding(model_name="meta/llama-3.3-70b-instruct", api_key="")
    assert evaluator.get_model_name() == "meta/llama-3.3-70b-instruct"
    
    test_gen = evaluator.generate("Test prompt furthermore crucial")
    assert isinstance(test_gen, str)
    assert len(test_gen) > 0


def test_deepeval_congruency_metric_evaluation():
    congruence_eval = DeepEvalCongruencyEvaluator(model_name="meta/llama-3.3-70b-instruct", api_key="")
    
    res = congruence_eval.evaluate_test_case(
        masked_input="Artificial intelligence has revolutionized [MASK_1], playing a crucial role [MASK_2].",
        infilled_actual="Artificial intelligence has revolutionized modern technology, playing a crucial role in reshaping global industries.",
        original_expected="Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide.",
    )
    
    assert "deepeval_score" in res
    assert "deepeval_reason" in res
    assert "framework" in res
    assert res["framework"] == "DeepEval GEval"
    assert res["deepeval_score_percent"] >= 0.0


def test_detector_with_deepeval_framework():
    client = NvidiaNIMClient(api_key="")
    detector = ClozeCongruenceDetector(nim_client=client)
    text = (
        "Furthermore, artificial intelligence plays a crucial role in modern computational paradigms. "
        "Moreover, navigating the multifaceted landscape of deep learning fosters transformative breakthroughs."
    )
    res = detector.analyze(text, mask_rate=0.30, num_passes=1)
    
    assert "deepeval_evaluation" in res
    assert "geval_score" in res["deepeval_evaluation"]
    assert "reason" in res["deepeval_evaluation"]
    assert res["deepeval_evaluation"]["framework"] == "DeepEval GEval"
