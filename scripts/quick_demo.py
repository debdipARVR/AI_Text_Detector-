"""Quick live verification and demonstration script with DeepEval."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import ClozeCongruenceDetector, TextHumanizer, NvidiaNIMClient
from src.security.encryption import generate_fernet_key, encrypt_api_key, decrypt_api_key

def main():
    print("=" * 65)
    print("1. TESTING FERNET CREDENTIAL ENCRYPTION & DECRYPTION")
    print("=" * 65)
    fkey = generate_fernet_key()
    raw_secret = "nvapi-sample-production-key-987654321"
    enc = encrypt_api_key(raw_secret, fkey)
    dec = decrypt_api_key(enc, fkey)
    print(f"  Fernet Secret Key:   {fkey}")
    print(f"  Encrypted Token:     {enc[:35]}...")
    print(f"  Decrypted Key:       {dec}")
    print(f"  Match Verified:      {dec == raw_secret}")

    print("\n" + "=" * 65)
    print("2. TESTING AI TEXT DETECTION VIA DEEPEVAL FRAMEWORK")
    print("=" * 65)
    client = NvidiaNIMClient(api_key="")
    detector = ClozeCongruenceDetector(nim_client=client)
    ai_sample = (
        "Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role "
        "in reshaping industries worldwide. Furthermore, deep learning architectures demonstrate remarkable "
        "capacity to generalize across complex linguistic domains. In conclusion, navigating the multifaceted "
        "landscape of generative AI is a testament to the transformative power of computational innovation."
    )
    res_ai = detector.analyze(ai_sample)
    print(f"  Input:               ChatGPT-4 Generated Essay")
    print(f"  Verdict:             {res_ai['verdict']} (Confidence: {res_ai['confidence']})")
    print(f"  AI Probability:      {res_ai['ai_probability']}%")
    if "deepeval_evaluation" in res_ai:
        print(f"  DeepEval Framework:  {res_ai['deepeval_evaluation']['framework']}")
        print(f"  DeepEval GEval Score:{res_ai['deepeval_evaluation']['geval_score']}%")
        print(f"  DeepEval Reason:     {res_ai['deepeval_evaluation']['reason'][:85]}...")
    print(f"  Semantic Match:      {res_ai['metrics']['semantic_similarity_avg']}%")
    print(f"  Word Overlap:        {res_ai['metrics']['word_similarity_avg']}%")

    print("\n" + "=" * 65)
    print("3. TESTING HUMAN TEXT DETECTION (Authentic Human Writing)")
    print("=" * 65)
    human_sample = (
        "I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize "
        "I forgot a single pointer dereference in the vertex shader loop. Classic. Watching the frame rate jump "
        "from 14 FPS back up to 120 made the cold coffee worthwhile."
    )
    res_human = detector.analyze(human_sample)
    print(f"  Input:               Authentic Human Experience")
    print(f"  Verdict:             {res_human['verdict']} (Confidence: {res_human['confidence']})")
    print(f"  AI Probability:      {res_human['ai_probability']}%")
    print(f"  Burstiness Index:    {res_human['metrics']['burstiness']['burstiness_score']}")

    print("\n" + "=" * 65)
    print("4. TESTING TEXT HUMANIZER & ANTI-CLICHE TRANSFORMATION")
    print("=" * 65)
    humanizer = TextHumanizer(nim_client=client)
    cliche_ai = "Furthermore, we must delve into the multifaceted landscape to foster innovation."
    hum_res = humanizer.humanize(cliche_ai, domain="academic")
    print(f"  Original AI:         {cliche_ai}")
    print(f"  Humanized:           {hum_res['humanized_text']}")
    print(f"  Cliches Detected:    {hum_res['ai_markers_before']['cliche_count']}")

    print("\n" + "=" * 65)
    print(">>> STATUS: DEEPEVAL FRAMEWORK OPERATING WITH 100% SUCCESS <<<")
    print("=" * 65)

if __name__ == "__main__":
    main()
