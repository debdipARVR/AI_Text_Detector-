"""Test live NVIDIA NIM models with the API key."""

import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.security.encryption import get_nvidia_api_key

def main():
    api_key = get_nvidia_api_key()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=30.0)
    
    test_models = ["z-ai/glm-5.2", "meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct", "thinkingmachines/inkling"]
    
    for m in test_models:
        print(f"Testing model: {m}...")
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Say 'hello' in 1 word."}],
                max_tokens=10,
            )
            print(f"  [SUCCESS] {m} responded: {res.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"  [ERROR] {m} failed: {e}")

if __name__ == "__main__":
    main()
