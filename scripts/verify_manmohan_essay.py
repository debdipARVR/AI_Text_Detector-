"""Verification script for Manmohan Singh essay evaluation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

essay = """
Dr. Manmohan Singh was one of the most respected economists and political leaders in modern Indian history. Known for his calm personality, intellectual honesty, and understated style of leadership, he served as the Prime Minister of India from 2004 to 2014. Unlike many politicians who are remembered for powerful speeches and dramatic political personalities, Manmohan Singh was known primarily for his knowledge, integrity, and quiet determination. His life represents the journey of a scholar and economist who rose to the highest political office in the country.

Manmohan Singh was born on 26 September 1932 in Gah, which was then part of undivided India and is now in Pakistan. The Partition of India in 1947 profoundly affected his early life, as his family moved to India. Despite difficult circumstances, he pursued education with exceptional dedication. He studied economics at Panjab University and later went to the University of Cambridge, where he completed his higher studies. He subsequently earned a doctorate in economics from the University of Oxford. His academic achievements established him as a distinguished economist long before he entered active politics.

Singh held several important positions in India's economic administration. He worked with the United Nations and served in various senior positions in the Government of India. He also became the Governor of the Reserve Bank of India and later served as India's Finance Minister. It was as Finance Minister in 1991 that he became particularly important to India's economic history.
"""

def main():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)
    res = detector.analyze(essay)

    print("=" * 70)
    print(" MANMOHAN SINGH ESSAY: KEY-VALUE PAIRED TWO-PASS EVALUATION")
    print("=" * 70)
    print(f"Final Verdict:              {res['verdict']} (Confidence: {res['confidence']})")
    print(f"Combined AI Probability:    {res['ai_probability']}%")
    print(f"Combined Congruence Score:  {res['combined_congruence_score']}%\n")

    print(f"[Pass 1 Results - 1 sentence per 4 lines]:")
    print(f"  Congruence Score:         {res['pass_1']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_1']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_1']['semantic_cosine']}%")
    print(f"  Semantic (10%):           {res['pass_1'].get('semantic_similarity', 50.0)}%")
    print(f"  Lexical Overlap (10%):    {res['pass_1']['lexical_similarity']}%")
    for s in res['pass_1']['spans']:
        print(f"  - Key {s['placeholder']}:")
        print(f"      Original:  \"{s['original_sentence']}\"")
        print(f"      AI Infill: \"{s['predicted_sentence']}\"")
        print(f"      Score:     Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}%")

    print(f"\n[Pass 2 Results - Alternate sentences every 2 lines]:")
    print(f"  Congruence Score:         {res['pass_2']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_2']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_2']['semantic_cosine']}%")
    for s in res['pass_2']['spans']:
        print(f"  - Key {s['placeholder']}:")
        print(f"      Original:  \"{s['original_sentence']}\"")
        print(f"      AI Infill: \"{s['predicted_sentence']}\"")
        print(f"      Score:     Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
