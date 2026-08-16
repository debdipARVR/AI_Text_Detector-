"""Unit and Integration tests for Two-Pass ClozeCongruenceDetector."""

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient


def test_detector_empty_text():
    detector = ClozeCongruenceDetector(nim_client=NvidiaNIMClient(api_key=""))
    res = detector.analyze("")
    assert res["ai_probability"] == 0.0
    assert "empty" in res.get("error", "").lower()


def test_detector_with_mock_client():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    
    text = (
        "Furthermore, artificial intelligence plays a crucial role in modern technological ecosystems. "
        "Moreover, it is a testament to human innovation across multiple domains. "
        "In conclusion, navigating the complexities of machine learning fosters transformative breakthroughs."
    )
    
    res = detector.analyze(text, model_name="z-ai/glm-5.2")
    assert "ai_probability" in res
    assert "verdict" in res
    assert "combined_congruence_score" in res
    assert "pass_1" in res
    assert "pass_2" in res
    assert "meaning_similarity" in res["pass_1"]
    assert "semantic_cosine" in res["pass_1"]
    assert "congruence_score" in res["pass_1"]
    assert "meaning_similarity" in res["pass_2"]
    assert "semantic_cosine" in res["pass_2"]
    assert "congruence_score" in res["pass_2"]
    assert "highlighted_html" in res
    assert "deepeval_evaluation" in res
