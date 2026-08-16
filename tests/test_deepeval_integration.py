"""Unit and integration tests for DeepEval framework metrics and custom evaluator."""

from src.deepeval_model_connect import NvidiaLLM_Understanding, DeepEvalCongruencyEvaluator
from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient


def test_deepeval_evaluator_creation():
    # Test with z-ai/glm-5.2 and thinkingmachines/inkling
    evaluator_glm = NvidiaLLM_Understanding(model_name="z-ai/glm-5.2", api_key="")
    assert evaluator_glm.get_model_name() == "z-ai/glm-5.2"
    
    evaluator_inkling = NvidiaLLM_Understanding(model_name="thinkingmachines/inkling", api_key="")
    assert evaluator_inkling.get_model_name() == "thinkingmachines/inkling"
    
    test_gen = evaluator_glm.generate("Test prompt furthermore crucial")
    assert isinstance(test_gen, str)
    assert len(test_gen) > 0


def test_deepeval_congruency_metric_evaluation():
    congruence_eval = DeepEvalCongruencyEvaluator(model_name="z-ai/glm-5.2", api_key="")
    
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
    assert res["evaluator_model"] == "z-ai/glm-5.2"


def test_detector_with_deepeval_framework():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    text = (
        "Furthermore, artificial intelligence plays a crucial role in modern computational paradigms. "
        "Moreover, navigating the multifaceted landscape of deep learning fosters transformative breakthroughs."
    )
    res = detector.analyze(text, mask_rate=0.30, num_passes=1, model_name="z-ai/glm-5.2")
    
    assert "deepeval_evaluation" in res
    assert "geval_score" in res["deepeval_evaluation"]
    assert "reason" in res["deepeval_evaluation"]
    assert res["deepeval_evaluation"]["framework"] == "DeepEval GEval"
    assert res["parameters"]["model_name"] == "z-ai/glm-5.2"
