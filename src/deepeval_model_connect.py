import os
from dotenv import load_dotenv
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM



load_dotenv()

class NvidiaLLM(DeepEvalBaseLLM):
    """Custom Deepeval evaluator backed by Nvidia NIM APIs"""
    """Since the DeepEvalBaseLLM is abstract class we need to implement all its methods"""

    def __init__(self, model_name : str= "sarvamai/sarvam-m"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_NIM_API_KEY"]

        )

    def load_model(self):
        self.client

    def generate(self, prompt: str):
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    

    async def a_generate(self, prompt: str):
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.model_name
    
