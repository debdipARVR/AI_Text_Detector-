"""Test passage-level bipartite optimal alignment vs strict positional alignment."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.metrics import compute_meaning_similarity, compute_cosine_similarity

orig_block = [
    "He studied at Harrow School, one of the most prestigious schools in Britain.",
    "Later, he joined Trinity College, Cambridge, where he studied Natural Sciences.",
    "After completing his graduation, he enrolled at the Inner Temple, London, to study law and qualified as a barrister."
]

pred_block = [
    "He completed his early schooling at Harrow before attending Trinity College, Cambridge to study Natural Sciences.",
    "He subsequently studied law at the Inner Temple in London and was called to the bar.",
    "During his stay in England, he was exposed to liberal ideas, democratic values, and modern governance."
]

def evaluate_bipartite(origs, preds):
    # For each original sentence, find the best matching prediction in the block
    matches = []
    for orig in origs:
        best_score = 0.0
        best_pred = ""
        for pred in preds:
            m = compute_meaning_similarity(orig, pred)
            c = compute_cosine_similarity(orig, pred)
            score = (0.40 * m) + (0.40 * c) + 0.20 * 0.6
            if score > best_score:
                best_score = score
                best_pred = pred
        matches.append((orig, best_pred, best_score))
    
    avg_score = sum(s for _, _, s in matches) / max(1, len(matches))
    return matches, avg_score

def main():
    matches, score = evaluate_bipartite(orig_block, pred_block)
    print("=== BIPARTITE ALIGNMENT RESULTS ===")
    for orig, pred, s in matches:
        print(f"Orig: \"{orig[:45]}...\"")
        print(f"Match: \"{pred[:45]}...\" -> Score: {round(s*100, 1)}%\n")
    print(f"Overall Block Congruence: {round(score * 100, 1)}%")

if __name__ == "__main__":
    main()
