"""Demonstrate Cloze Congruence scores with Pass 2 and Pass 3 on AI vs Human Text."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

ai_essay = """
Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide. 
Furthermore, deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains. 
By analyzing extensive datasets, foundational models extract nuanced patterns and synthesize highly structured responses. 
In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, fostering continuous advancements across science and society.
"""

human_text = """
I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize I forgot a single pointer dereference in the vertex shader loop. 
Classic dev mistake. 
You'd think after ten years in game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. 
Still, watching the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee entirely worthwhile.
"""

def run():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)

    print("=" * 80)
    print(" 1. EVALUATION OF CANONICAL AI TEXT (Pass 2 & Pass 3)")
    print("=" * 80)
    ai_res = detector.analyze(ai_essay)
    print(f"Final Verdict:              {ai_res['verdict']} (Confidence: {ai_res['confidence']})")
    print(f"Combined AI Probability:    {ai_res['ai_probability']}%")
    print(f"Pass 2 (Alternate):         {ai_res['pass_2']['congruence_score']}% (Meaning: {ai_res['pass_2']['meaning_similarity']}%, Cosine: {ai_res['pass_2']['semantic_cosine']}%)")
    print(f"Pass 3 (Middle 3-Sentence): {ai_res['pass_3']['congruence_score']}% (Meaning: {ai_res['pass_3']['meaning_similarity']}%, Cosine: {ai_res['pass_3']['semantic_cosine']}%)")

    print("\n" + "=" * 80)
    print(" 2. EVALUATION OF HUMAN TEXT (Pass 2 & Pass 3)")
    print("=" * 80)
    hum_res = detector.analyze(human_text)
    print(f"Final Verdict:              {hum_res['verdict']} (Confidence: {hum_res['confidence']})")
    print(f"Combined AI Probability:    {hum_res['ai_probability']}%")
    print(f"Pass 2 (Alternate):         {hum_res['pass_2']['congruence_score']}% (Meaning: {hum_res['pass_2']['meaning_similarity']}%, Cosine: {hum_res['pass_2']['semantic_cosine']}%)")
    print(f"Pass 3 (Middle 3-Sentence): {hum_res['pass_3']['congruence_score']}% (Meaning: {hum_res['pass_3']['meaning_similarity']}%, Cosine: {hum_res['pass_3']['semantic_cosine']}%)")
    print("=" * 80)

if __name__ == "__main__":
    run()
