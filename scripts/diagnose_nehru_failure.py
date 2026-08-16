"""Diagnose exact reasons why the Nehru essay scored 40% and test solutions."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.cloze_masker import ClozeMasker
from src.engine.metrics import compute_meaning_similarity, compute_cosine_similarity, compute_semantic_congruence

# Example Paragraph 3 from the Nehru essay:
p3_original = [
    "At the age of fifteen, Nehru went to England for higher studies.",
    "He studied at Harrow School, one of the most prestigious schools in Britain.",
    "Later, he joined Trinity College, Cambridge, where he studied Natural Sciences.",
    "After completing his graduation, he enrolled at the Inner Temple, London, to study law and qualified as a barrister.",
    "These experiences greatly influenced his thinking and later shaped his vision for India."
]

# What an LLM typically predicts when infilled:
p3_llm_predictions = [
    "He completed his early schooling at Harrow before attending Trinity College, Cambridge to study Natural Sciences.",
    "He subsequently studied law at the Inner Temple in London and was called to the bar.",
    "During his years in Britain, he was deeply exposed to Western political thought and democratic ideals."
]

def diagnose():
    print("=" * 80)
    print(" DIAGNOSTIC ANALYSIS: WHY CLOZE INFILLING DIVERGES ON BIOGRAPHICAL TEXTS")
    print("=" * 80)
    print("1. Paragraph Context:")
    print(f"   Opening Anchor: \"{p3_original[0]}\"")
    print(f"   Closing Anchor: \"{p3_original[4]}\"\n")

    print("2. Ground Truth Middle Sentences:")
    for i, s in enumerate(p3_original[1:4]):
        print(f"   Orig [{i+1}]: \"{s}\"")

    print("\n3. Plausible LLM Infill Completions:")
    for i, s in enumerate(p3_llm_predictions):
        print(f"   Pred [{i+1}]: \"{s}\"")

    print("\n4. Metric Comparisons between Original and LLM Infill:")
    for i in range(3):
        orig = p3_original[i+1]
        pred = p3_llm_predictions[i]
        meaning = compute_meaning_similarity(orig, pred)
        cosine = compute_cosine_similarity(orig, pred)
        semantic = compute_semantic_congruence(orig, pred)
        comp = (0.40 * meaning) + (0.40 * cosine) + (0.10 * semantic) + (0.10 * 0.3)
        print(f"   - Sentence {i+1}: Meaning={round(meaning*100,1)}% | Cosine={round(cosine*100,1)}% | Composite={round(comp*100,1)}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose()
