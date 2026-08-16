"""Unit tests for TextHumanizer."""

from src.engine.humanizer import TextHumanizer, HUMANIZER_MODES


def test_generate_humanize_prompt():
    humanizer = TextHumanizer()
    for mode in HUMANIZER_MODES.keys():
        bundle = humanizer.generate_humanize_prompt(domain=mode)
        assert bundle["domain"] == mode
        assert "system_prompt" in bundle
        assert "user_prompt_template" in bundle
        assert "Burstiness" in bundle["user_prompt_template"]


def test_analyze_ai_markers():
    humanizer = TextHumanizer()
    text = "In conclusion, it is a testament to the crucial role of AI in navigating the complexities of this landscape."
    analysis = humanizer.analyze_ai_markers(text)
    assert analysis["cliche_count"] >= 2
    assert any("testament" in c["phrase"].lower() for c in analysis["cliches_detected"])


def test_heuristic_humanize():
    humanizer = TextHumanizer()
    text = "Furthermore, we must delve into the multifaceted landscape."
    res = humanizer.humanize(text, domain="academic")
    assert "humanized_text" in res
    assert len(res["humanized_text"]) > 0
