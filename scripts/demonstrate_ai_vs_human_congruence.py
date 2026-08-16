"""Demonstrate Cloze Congruence scores on AI-Generated Text vs Human-Authored Text."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

ai_generated_essay = """
Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide. 
Furthermore, deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains. 
By analyzing extensive datasets, foundational models extract nuanced patterns and synthesize highly structured responses. 
In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, fostering continuous advancements across science and society.
"""

human_authored_text = """
I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize I forgot a single pointer dereference in the vertex shader loop. 
Classic dev mistake. 
You'd think after ten years in game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. 
Still, watching the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee entirely worthwhile.
"""

def run_comparison():
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)

    print("=" * 80)
    print(" 1. EVALUATION OF CANONICAL AI-GENERATED TEXT (GPT-4 / GLM-5.2 Style)")
    print("=" * 80)
    ai_res = detector.analyze(ai_generated_essay)
    print(f"Final Verdict:              {ai_res['verdict']} (Confidence: {ai_res['confidence']})")
    print(f"Combined AI Probability:    {ai_res['ai_probability']}%")
    print(f"Combined Congruence Score:  {ai_res['combined_congruence_score']}%\n")
    print(f"Pass 1 Congruence (Sparse): {ai_res['pass_1']['congruence_score']}% (Meaning: {ai_res['pass_1']['meaning_similarity']}%, Cosine: {ai_res['pass_1']['semantic_cosine']}%)")
    print(f"Pass 2 Congruence (Alt):    {ai_res['pass_2']['congruence_score']}% (Meaning: {ai_res['pass_2']['meaning_similarity']}%, Cosine: {ai_res['pass_2']['semantic_cosine']}%)")
    for s in ai_res['pass_1']['spans']:
        print(f"  - Key {s['placeholder']}:")
        print(f"      Original:  \"{s['original_sentence']}\"")
        print(f"      AI Infill: \"{s['predicted_sentence']}\"")
        print(f"      Scores:    Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}% | Status={s['status']}")

    print("\n" + "=" * 80)
    print(" 2. EVALUATION OF HUMAN-AUTHORED TEXT (Personal Narrative)")
    print("=" * 80)
    hum_res = detector.analyze(human_authored_text)
    print(f"Final Verdict:              {hum_res['verdict']} (Confidence: {hum_res['confidence']})")
    print(f"Combined AI Probability:    {hum_res['ai_probability']}%")
    print(f"Combined Congruence Score:  {hum_res['combined_congruence_score']}%\n")
    print(f"Pass 1 Congruence (Sparse): {hum_res['pass_1']['congruence_score']}% (Meaning: {hum_res['pass_1']['meaning_similarity']}%, Cosine: {hum_res['pass_1']['semantic_cosine']}%)")
    print(f"Pass 2 Congruence (Alt):    {hum_res['pass_2']['congruence_score']}% (Meaning: {hum_res['pass_2']['meaning_similarity']}%, Cosine: {hum_res['pass_2']['semantic_cosine']}%)")
    for s in hum_res['pass_1']['spans']:
        print(f"  - Key {s['placeholder']}:")
        print(f"      Original:  \"{s['original_sentence']}\"")
        print(f"      AI Infill: \"{s['predicted_sentence']}\"")
        print(f"      Scores:    Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}% | Status={s['status']}")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
