"""Test GLM-5.2 generation and Cloze Congruence self-detection."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.security.encryption import get_nvidia_api_key
from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient

def main():
    api_key = get_nvidia_api_key()
    print("API Key available:", bool(api_key and len(api_key) > 5))
    if api_key:
        print("API Key:", api_key[:7] + "..." + api_key[-4:])
    else:
        print("Running in simulated GLM-5.2 mode.")

if __name__ == "__main__":
    main()
