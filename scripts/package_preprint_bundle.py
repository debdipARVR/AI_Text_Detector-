"""Package complete arXiv preprint submission bundle."""

import os
import shutil
import zipfile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATEX_DIR = os.path.join(BASE_DIR, "latex")
OUT_DIR = os.path.join(BASE_DIR, "data")
ZIP_PATH = os.path.join(OUT_DIR, "arxiv_preprint_bundle.zip")

# Ensure destination exists
os.makedirs(OUT_DIR, exist_ok=True)

# Create zip bundle
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. paper.tex
    tex_path = os.path.join(LATEX_DIR, "paper.tex")
    if os.path.exists(tex_path):
        zipf.write(tex_path, arcname="paper.tex")
        print("Added paper.tex")

    # 2. references.bib
    bib_path = os.path.join(LATEX_DIR, "references.bib")
    if os.path.exists(bib_path):
        zipf.write(bib_path, arcname="references.bib")
        print("Added references.bib")

    # 3. Add benchmark results summary
    bm_path = os.path.join(OUT_DIR, "benchmark_results", "benchmark_results.json")
    if os.path.exists(bm_path):
        zipf.write(bm_path, arcname="data/benchmark_results.json")
        print("Added benchmark_results.json")

    # 4. Instructions
    readme_content = """# arXiv / TechRxiv Preprint Submission Bundle

## Paper Title:
Multi-Pass Sentence Cloze Infilling with Sigmoidal Semantic Congruence for Robust AI Text Detection

## Files Included:
- `paper.tex`: Main LaTeX source document (IEEEtran format).
- `references.bib`: Complete BibTeX bibliography with citations for DetectGPT, Fast-DetectGPT, Ghostbuster, DeepEval, and Watermarking.
- `data/benchmark_results.json`: Empirical evaluation logs across 6 academic domains.

## How to Submit:
1. **Overleaf**: Go to https://overleaf.com -> 'New Project' -> 'Upload Project' -> Upload `arxiv_preprint_bundle.zip`.
2. **arXiv (cs.CL / cs.AI)**: Go to https://arxiv.org/submit -> Upload `paper.tex` and `references.bib`.
3. **TechRxiv**: Upload PDF compiled from Overleaf to https://www.techrxiv.org.
"""
    zipf.writestr("README.md", readme_content)
    print("Added README.md")

print(f"\nSuccessfully created preprint zip bundle at: {ZIP_PATH} ({round(os.path.getsize(ZIP_PATH)/1024, 2)} KB)")
