# ClozeCongruence: AI Text Detector & Humanization Engine 🕵️‍♂️⚡

[![CI Test Suite](https://github.com/debdipARVR/AI_Text_Detector-/actions/workflows/ci.yml/badge.svg)](https://github.com/debdipARVR/AI_Text_Detector-/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM%20API-76B900.svg)](https://build.nvidia.com/)
[![Security](https://img.shields.io/badge/Security-Fernet%20AES--128-green.svg)](https://cryptography.io/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

A statistical AI text detection and humanization platform powered by **Randomized Cloze Masking & Infilling Congruence**, **NVIDIA NIM LLMs**, **Fernet Credential Encryption**, and a **Modern Interactive Web Playground**.

---

## 📌 The Detection Principle: Cloze Congruence

Traditional perplexity detectors often suffer from high false-positive rates on academic or structured human writing. **ClozeCongruence** approaches detection through bidirectional contextual predictability:

```
                      ┌──────────────────────────────────────┐
                      │          Input Text Document         │
                      └──────────────────┬───────────────────┘
                                         │
                   1. Randomized Sentence & Clause Masking (30%)
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │ "AI models [MASK_1] and synthesize   │
                      │  highly structured [MASK_2]..."      │
                      └──────────────────┬───────────────────┘
                                         │
                   2. Infill via High-Capacity NVIDIA NIM LLM
                      (e.g., LLaMA-3.3-70B, Nemotron-70B)
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │ Predicted:                           │
                      │ [MASK_1] -> "extract nuanced patterns"│
                      │ [MASK_2] -> "responses"              │
                      └──────────────────┬───────────────────┘
                                         │
                   3. Multi-Dimensional Congruence Scoring
                      (Semantic + Lexical + Burstiness)
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
       High Congruence (>= 72%)                      Low Congruence (< 40%)
   ---------------------------------             -------------------------------
    LLM accurately reproduced original            LLM diverged from original text
        👉 LIKELY AI-GENERATED                         👉 LIKELY HUMAN-AUTHORED
```

### Key Linguistic Insights:
1. **Low Entropy in AI Text**: LLMs choose high-probability token sequences with stereotypical syntactic pathways. When masked, another LLM given the context easily regenerates nearly identical completions (**High Congruence**).
2. **Idiosyncratic Human Style**: Human writing exhibits high syntactic burstiness (spiky sentence lengths), unique metaphors, and non-templated phrasing. When masked, an LLM predicts generic completions that diverge significantly (**Low Congruence**).

---

## ✨ Features

- 🎯 **Randomized Cloze Infilling Congruence**: Multi-pass Monte Carlo clause and sentence masking.
- 🚀 **NVIDIA NIM Integration**: Connects to NVIDIA NIM endpoints (`meta/llama-3.3-70b-instruct`, `nvidia/llama-3.1-nemotron-70b-instruct`, `mistralai/mixtral-8x22b-instruct`, `sarvamai/sarvam-m`).
- 🔒 **Fernet Key Credential Encryption**: At-rest and in-transit credential protection using AES-128-CBC with HMAC authentication. Zero raw keys stored in Git.
- 🎨 **Modern Interactive Web Playground**:
  - Live Cloze Heatmap with interactive span tooltips (Original vs Infill diff).
  - Animated Radial Score Gauge & Metric Progress Bars.
  - 1-Click Test Presets (GPT-4 Essay, Human Academic, Mixed Article, Humanized).
- ✍️ **Humanizer Studio & Prompt Lab**:
  - Detects AI clichés ("delve", "testament", "crucial role", "landscape").
  - Rewrites text with high burstiness and organic cadence.
  - Before vs After benchmark scoring comparison.
  - Battle-tested Anti-Detection Prompt Generator.
- 💻 **Full CLI & REST API**: Run via command line, Python SDK, or FastAPI endpoints.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/debdipARVR/AI_Text_Detector-.git
cd AI_Text_Detector-
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔒 Fernet Credential Encryption Setup

To ensure your NVIDIA NIM API key is never committed in plaintext:

### Step 1: Encrypt Your Key
Run the built-in encryption tool:
```bash
python scripts/encrypt_credentials.py encrypt --api-key "nvapi-YOUR_ACTUAL_NVIDIA_API_KEY"
```

Output:
```
=======================================================
 NVIDIA NIM API Key Encrypted Successfully
=======================================================
Masked Original API Key: nvapi...xxxx
FERNET_SECRET_KEY=Vpko6Uhuq-6Kmpt9Ue3cte5RelO4ZfiZal8c69fxAcM=
FERNET_ENCRYPTED_NVIDIA_API_KEY=gAAAAABn...encrypted_token...
=======================================================
```

### Step 2: Configure Local `.env`
Create a `.env` file in the project root (this file is automatically ignored by `.gitignore`):
```env
FERNET_SECRET_KEY=Vpko6Uhuq-6Kmpt9Ue3cte5RelO4ZfiZal8c69fxAcM=
FERNET_ENCRYPTED_NVIDIA_API_KEY=gAAAAABn...encrypted_token...
```

### Step 3: Configure GitHub Actions Secrets
In your GitHub repository:
1. Go to **Settings** ➔ **Secrets and variables** ➔ **Actions**
2. Add **`FERNET_SECRET_KEY`** with your secret key
3. Add **`FERNET_ENCRYPTED_NVIDIA_API_KEY`** with your encrypted token

---

## 🚀 Running the Web Playground

Launch the local interactive playground server:
```bash
python run_playground.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

- **Interactive UI**: `http://127.0.0.1:8000`
- **Swagger / OpenAPI Docs**: `http://127.0.0.1:8000/docs`

> *Note: If no API key is provided, the detector seamlessly runs in Offline Simulation Fallback mode for immediate exploration.*

---

## 💻 CLI Usage

### 1. Run Detection
```bash
python -m src.cli detect "Artificial intelligence plays a crucial role in modern technological ecosystems. Furthermore, it is a testament to human innovation."
```

### 2. Humanize Text
```bash
python -m src.cli humanize "Furthermore, we must delve into the multifaceted landscape." --domain academic
```

### 3. Generate Anti-Detection Prompt
```bash
python -m src.cli prompt --domain academic --audience "Peer Reviewers"
```

---

## 🧪 Running Automated Tests

Run the complete test suite across all modules:
```bash
pytest tests/ -v
```

All 31 unit, integration, and security test cases validate:
- Fernet key encryption/decryption cycles & token validation.
- Cloze segmentation, random masking rates, and multi-pass consistency.
- Lexical, semantic, and burstiness scoring formulas.
- End-to-end detector pipeline with live & mock clients.
- FastAPI REST endpoints and static asset serving.

---

## 📂 Project Structure

```
AI_Text_Detector/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI with encrypted secret variables
├── scripts/
│   └── encrypt_credentials.py  # CLI runner for Fernet key encryption
├── src/
│   ├── security/               # Fernet encryption and credential manager
│   │   ├── __init__.py
│   │   ├── encryption.py
│   │   └── cli.py
│   ├── engine/                 # Core Cloze Congruence detection algorithms
│   │   ├── __init__.py
│   │   ├── cloze_masker.py     # Sentence & clause masking engine
│   │   ├── nim_client.py       # NVIDIA NIM OpenAI-compatible client
│   │   ├── metrics.py          # Lexical, semantic & burstiness metrics
│   │   ├── detector.py         # Master ClozeCongruenceDetector
│   │   └── humanizer.py        # Text humanizer & prompt generator
│   ├── web/                    # FastAPI web application & playground
│   │   ├── __init__.py
│   │   ├── app.py              # REST API & static file routes
│   │   └── static/             # Modern HTML5/CSS3/JS frontend
│   │       ├── index.html
│   │       ├── app.css
│   │       └── app.js
│   └── cli.py                  # Master terminal CLI
├── tests/                      # Pytest test suite (31 tests)
│   ├── test_cloze_masker.py
│   ├── test_detector.py
│   ├── test_encryption.py
│   ├── test_humanizer.py
│   ├── test_metrics.py
│   └── test_web_api.py
├── .gitignore                  # Strict security ignore rules (keys, .env, caches)
├── pyproject.toml              # Project metadata & build system
├── requirements.txt            # Dependency list
├── run_playground.py           # Web server runner
└── README.md                   # Project documentation
```

---

## 🤝 Contributing & Security

Contributions are welcome! Please ensure that:
1. No `.env`, secret keys, or Fernet credentials are ever committed.
2. All new features include corresponding unit tests in `tests/`.
3. `pytest tests/` passes before opening a pull request.

---

## 📜 License
MIT License. Created by [debdipARVR](https://github.com/debdipARVR).
