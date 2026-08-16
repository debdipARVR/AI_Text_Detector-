"""Unit and Integration tests for ClozeCongruenceDetector."""

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient


def test_detector_empty_text():
    detector = ClozeCongruenceDetector(nim_client=NvidiaNIMClient(api_key=""))
    res = detector.analyze("")
    assert res["ai_probability"] == 0.0
    assert "empty" in res.get("error", "").lower()


def test_detector_with_mock_client():
    client = NvidiaNIMClient(api_key="")
    detector = ClozeCongruenceDetector(nim_client=client, default_passes=1)
    
    text = (
        "Furthermore, artificial intelligence plays a crucial role in modern technological ecosystems. "
        "Moreover, it is a testament to human innovation across multiple domains. "
        "In conclusion, navigating the complexities of machine learning fosters transformative breakthroughs."
    )
    
    res = detector.analyze(text, mask_rate=0.30, num_passes=1)
    assert "ai_probability" in res
    assert "verdict" in res
    assert "metrics" in res
    assert "semantic_similarity_avg" in res["metrics"]
    assert "word_similarity_avg" in res["metrics"]
    assert "spans" in res
    assert len(res["spans"]) > 0
    assert "highlighted_html" in res
    assert "deepeval_evaluation" in res
