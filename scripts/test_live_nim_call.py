import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.nim_client import NvidiaNIMClient
from src.engine.cloze_masker import ClozeMasker

client = NvidiaNIMClient()
masker = ClozeMasker()

sample_passage = "Quantum computers leverage superposition to perform computations exponentially faster. Classical computers rely on deterministic binary bits. This fundamental difference unlocks unprecedented potential for cryptography and materials science."

masked = masker.mask_pass_2(sample_passage)
print("Masked Text:\n", masked.masked_text)

preds = client.infill_cloze_spans(masked.masked_text, masked.spans, model_name="z-ai/glm-5.2")
print("\nNVIDIA NIM Infilled Predictions (GLM-5.2):")
for placeholder, pred in preds.items():
    print(f"  {placeholder} -> '{pred}'")
