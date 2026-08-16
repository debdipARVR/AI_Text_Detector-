"""Generate text using live z-ai/glm-5.2 from NVIDIA NIM and run Two-Pass Cloze Congruence detection."""

import json
import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient
from src.security.encryption import get_nvidia_api_key

def main():
    api_key = get_nvidia_api_key()
    if not api_key:
        print("Error: No NVIDIA NIM API key found.")
        return

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=45.0)
    model = "z-ai/glm-5.2"

    print(f"1. Prompting {model} to generate an original essay on Jawaharlal Nehru...")
    prompt = (
        "Write a detailed 4-paragraph biographical essay about Jawaharlal Nehru, "
        "highlighting his role as a statesman, his educational background, his democratic vision, and his industrial development policies."
    )

    gen_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional historian and essayist."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=800,
    )

    generated_essay = gen_response.choices[0].message.content or ""
    print("\n--- GENERATED ESSAY BY GLM-5.2 ---")
    print(generated_essay[:400] + "...\n")

    print(f"2. Running Two-Pass Cloze Congruence Detection using {model} with live infilling and DeepEval...")
    nim_client = NvidiaNIMClient(api_key=api_key, default_model=model)
    detector = ClozeCongruenceDetector(nim_client=nim_client)
    res = detector.analyze(generated_essay, model_name=model)

    print("=" * 80)
    print(" LIVE GLM-5.2 GENERATION & SELF-DETECTION RESULTS")
    print("=" * 80)
    print(f"Final Verdict:              {res['verdict']} (Confidence: {res['confidence']})")
    print(f"Combined AI Probability:    {res['ai_probability']}%")
    print(f"Combined Congruence Score:  {res['combined_congruence_score']}%\n")

    print(f"[Pass 1: Sparse] ({res['pass_1']['sentences_masked_count']} blanks):")
    print(f"  Congruence Score:         {res['pass_1']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_1']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_1']['semantic_cosine']}%")
    print(f"  Semantic Alignment (10%): {res['pass_1'].get('semantic_similarity', 50.0)}%")
    print(f"  Lexical Overlap (10%):    {res['pass_1']['lexical_similarity']}%")
    print("  Sample Infill Pair:")
    if res['pass_1']['spans']:
        s = res['pass_1']['spans'][0]
        print(f"    - Key {s['placeholder']}:")
        print(f"        Original:  \"{s['original_sentence']}\"")
        print(f"        AI Infill: \"{s['predicted_sentence']}\"")
        print(f"        Scores:    Meaning={s['meaning_similarity']}% | Cosine={s['semantic_cosine']}% | Congruence={s['congruence']}%")

    print(f"\n[Pass 2: Alternate] ({res['pass_2']['sentences_masked_count']} blanks):")
    print(f"  Congruence Score:         {res['pass_2']['congruence_score']}%")
    print(f"  Meaning Similarity (40%): {res['pass_2']['meaning_similarity']}%")
    print(f"  Semantic Cosine (40%):    {res['pass_2']['semantic_cosine']}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
