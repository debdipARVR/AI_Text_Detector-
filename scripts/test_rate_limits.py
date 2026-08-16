"""Test multiple models for rate limit availability."""

import os
import sys
import time
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.security.encryption import get_nvidia_api_key

def main():
    api_key = get_nvidia_api_key()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=20.0)
    
    candidate_models = [
        "z-ai/glm-5.2",
        "thinkingmachines/inkling",
        "meta/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "google/gemma-3-12b-it",
    ]
    
    for m in candidate_models:
        print(f"Testing {m}...")
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Write one short sentence about Nehru."}],
                max_tokens=30,
            )
            print(f"  -> SUCCESS ({m}): {res.choices[0].message.content.strip()}")
            break
        except Exception as e:
            print(f"  -> FAILED ({m}): {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
