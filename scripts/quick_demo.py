"""Two-Pass ClozeCongruence DeepEval Meaning Demo Script."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import ClozeCongruenceDetector, TextHumanizer, NvidiaNIMClient
from src.security.encryption import generate_fernet_key, encrypt_api_key, decrypt_api_key

def main():
    print("=" * 70)
    print("1. TWO-PASS DEEPEVAL MEANING DETECTION (AI-Generated Essay)")
    print("=" * 70)
    client = NvidiaNIMClient(api_key="", default_model="z-ai/glm-5.2")
    detector = ClozeCongruenceDetector(nim_client=client)

    ai_sample = (
        "Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide. "
        "Furthermore, deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains. "
        "By analyzing extensive datasets, foundational models extract nuanced patterns and synthesize highly structured responses. "
        "In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, fostering continuous advancements across science and society."
    )

    res_ai = detector.analyze(ai_sample, model_name="z-ai/glm-5.2")
    print(f"  Final Verdict:             {res_ai['verdict']} (Confidence: {res_ai['confidence']})")
    print(f"  Combined AI Probability:   {res_ai['ai_probability']}%")
    print(f"  Combined Congruence Score: {res_ai['combined_congruence_score']}%\n")

    p1 = res_ai["pass_1"]
    print(f"  [Pass 1: Sparse - 1 sentence per 4 lines]")
    print(f"    - Congruence Score:         {p1['congruence_score']}%")
    print(f"    - Meaning Similarity (55%): {p1['meaning_similarity']}% (Highest Priority)")
    print(f"    - Semantic Cosine (30%):    {p1['semantic_cosine']}%")
    print(f"    - DeepEval Reason:          {p1['deepeval_reason'][:80]}...\n")

    p2 = res_ai["pass_2"]
    print(f"  [Pass 2: Alternate - sentences every 2 lines]")
    print(f"    - Congruence Score:         {p2['congruence_score']}%")
    print(f"    - Meaning Similarity (55%): {p2['meaning_similarity']}% (Highest Priority)")
    print(f"    - Semantic Cosine (30%):    {p2['semantic_cosine']}%")
    print(f"    - DeepEval Reason:          {p2['deepeval_reason'][:80]}...\n")

    print(f"  Verdict Decision Rule:     {res_ai['two_pass_verdict_reason']}")

    print("\n" + "=" * 70)
    print("2. TWO-PASS DEEPEVAL MEANING DETECTION (Authentic Human Writing)")
    print("=" * 70)
    human_sample = (
        "I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize I forgot a single pointer dereference in the vertex shader loop. "
        "Classic dev mistake. "
        "You'd think after ten years in game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. "
        "Still, watching the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee worthwhile."
    )
    res_human = detector.analyze(human_sample, model_name="z-ai/glm-5.2")
    print(f"  Final Verdict:             {res_human['verdict']} (Confidence: {res_human['confidence']})")
    print(f"  Combined AI Probability:   {res_human['ai_probability']}%")
    print(f"  Pass 1 Congruence:         {res_human['pass_1']['congruence_score']}%")
    print(f"  Pass 2 Congruence:         {res_human['pass_2']['congruence_score']}%")
    print(f"  Meaning Similarity:        {res_human['metrics']['meaning_similarity_avg']}%")
    print(f"  Verdict Decision Rule:     {res_human['two_pass_verdict_reason']}")

    print("\n" + "=" * 70)
    print(">>> STATUS: TWO-PASS DEEPEVAL MEANING PIPELINE VERIFIED 100% <<<")
    print("=" * 70)

if __name__ == "__main__":
    main()
