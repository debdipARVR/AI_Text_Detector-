"""DeepEval Custom LLM Evaluator backed by NVIDIA NIM APIs and Fernet Encryption."""

import os
from dotenv import load_dotenv
from openai import OpenAI

try:
    from deepeval.models import DeepEvalBaseLLM
except ImportError:
    # Graceful fallback if deepeval is not installed in the environment
    class DeepEvalBaseLLM:
        pass

from .security.encryption import get_nvidia_api_key

load_dotenv()


class NvidiaLLM_Understanding(DeepEvalBaseLLM):
    """Custom Deepeval evaluator backed by Nvidia NIM APIs with Fernet security support."""

    def __init__(self, model_name: str = "sarvamai/sarvam-m", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or get_nvidia_api_key() or "mock-key"
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


class NvidiaLLM_Embeding_Similarity:
    """NVIDIA NIM Embedding and Semantic Congruence Evaluator."""

    def __init__(self, model_name: str = "nvidia/nv-embedqa-e5-v5", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or get_nvidia_api_key() or "mock-key"
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )