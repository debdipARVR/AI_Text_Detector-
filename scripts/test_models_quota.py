import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.nim_client import NvidiaNIMClient

client = NvidiaNIMClient()

models_to_test = [
    "z-ai/glm-5.2",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "thinkingmachines/inkling"
]

for m in models_to_test:
    try:
        resp = client.client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"Model {m}: SUCCESS -> '{resp.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"Model {m}: FAILED -> {type(e).__name__}: {e}")
