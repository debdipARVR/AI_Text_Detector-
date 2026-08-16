"""List available model IDs from NVIDIA NIM API."""

import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.security.encryption import get_nvidia_api_key

def main():
    api_key = get_nvidia_api_key()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, timeout=15.0)
    try:
        models = client.models.list()
        print("Available NVIDIA NIM models:")
        for m in models.data:
            print(f" - {m.id}")
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    main()
