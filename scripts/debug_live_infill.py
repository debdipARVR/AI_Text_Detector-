import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.nim_client import NvidiaNIMClient
from src.engine.cloze_masker import ClozeMasker

client = NvidiaNIMClient()
masker = ClozeMasker()

sample_text = "The study of quantum computing is growing fast. [1] It allows exponential speedups for specialized problems."
mask = masker.mask_pass_2(sample_text)

try:
    res = client._infill_live(mask.masked_text, mask.spans, "z-ai/glm-5.2", 0.0)
    print("Direct _infill_live success:", res)
except Exception as e:
    import traceback
    print("Direct _infill_live ERROR:", type(e), e)
    traceback.print_exc()
