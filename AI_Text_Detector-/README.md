# AI_Text_Detector- 🕵️‍♂️

**A Python-based AI text detection pipeline designed to distinguish authentic human writing from LLM-generated content.**

## 📌 Overview
As Large Language Models (LLMs) become increasingly sophisticated, standard detection methods frequently suffer from high False Positive rates—incorrectly flagging highly structured human writing as AI. 

**CarbonTrace** is a Python toolkit built to analyze text using [insert your method: e.g., advanced stylometric analysis, perplexity/burstiness scoring, or a fine-tuned RoBERTa classifier]. It is specifically engineered to be benchmarked against uncontaminated, "air-gapped" human datasets to ensure reliable performance in real-world applications.

## 🚀 Features
* **Statistical Text Analysis:** Calculates sentence-level variance, burstiness, and linguistic predictability.
* **Low False-Positive Architecture:** Tuned specifically to recognize the nuanced imperfections and structural flow unique to human authorship.
* **Extensible Pipeline:** Easily integrate your own custom models or test against offline benchmark datasets.

## 🛠️ Installation

Requires Python 3.9+

```bash
git clone git@github.com:debdipARVR/AI_Text_Detector-.git

cd CarbonTrace
pip install poetry
poetry install
```

## 💻 Quick Start / Usage

```python
from carbontrace import Detector

# Initialize the detection engine
detector = Detector(model_path="models/base_config")

text_sample = """
Your text to analyze goes here.
"""

# Run analysis
results = detector.analyze(text_sample)

print(f"AI Probability: {results.ai_probability}%")
print(f"Burstiness Score: {results.burstiness}")
print(f"Perplexity: {results.perplexity}")
```

## 📊 Benchmarking and Data Integrity
To accurately evaluate this tool, **do not** benchmark it against standard, open-web essay datasets (as these are likely contaminated with AI or have been memorized by foundational models). 

We recommend testing `CarbonTrace` against offline, "clean-room" datasets (such as embargoed academic theses or physical transcriptions) to get a true measure of its False Positive Rate (FPR). 

*Note: Benchmark datasets are kept out of this repository to prevent web-scraper ingestion. Please place your local test data in the `.gitignore`'d `/data/` directory.*

## 🤝 Contributing
Contributions are welcome. Please ensure that any PRs involving model fine-tuning include documentation on the provenance of the training data used, verifying it is free from AI contamination.
