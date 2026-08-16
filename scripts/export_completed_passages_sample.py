"""Extract and showcase detailed AI reconstructed passages and infilled sentences."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.detector import ClozeCongruenceDetector
from src.engine.nim_client import NvidiaNIMClient
from scripts.evaluate_120_essays import build_essay_text

client = NvidiaNIMClient(api_key="")
detector = ClozeCongruenceDetector(nim_client=client)

# Select 3 representative essays
sample_topics = [
    (1, "Philosophy & Human Nature", "Is intelligence more valuable than wisdom?"),
    (11, "AI & Technology", "Will AI make humans intellectually stronger or weaker?"),
    (21, "Psychology", "Why do intelligent people make irrational decisions?"),
]

output_data = []

for essay_id, domain, title in sample_topics:
    text = build_essay_text(title)
    res = detector.analyze(text)
    output_data.append({
        "essay_id": essay_id,
        "domain": domain,
        "title": title,
        "pass_2": {
            "masked_text": res["pass_2"]["masked_text"],
            "reconstructed_text": res["pass_2"]["reconstructed_text"],
            "spans": res["pass_2"]["spans"],
            "congruence": res["pass_2"]["congruence_score"],
        },
        "pass_3": {
            "masked_text": res["pass_3"]["masked_text"],
            "reconstructed_text": res["pass_3"]["reconstructed_text"],
            "spans": res["pass_3"]["spans"],
            "congruence": res["pass_3"]["congruence_score"],
        }
    })

out_path = os.path.join("data", "benchmark_results", "sample_completed_passages.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved completed passages to {out_path}")
