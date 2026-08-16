"""Streamlit Web Application for ClozeCongruence AI Text Detector & Humanizer with DeepEval."""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.engine import (
    ClozeCongruenceDetector,
    TextHumanizer,
    NvidiaNIMClient,
    NVIDIA_MODELS,
    HUMANIZER_MODES,
)
from src.deepeval_model_connect import DeepEvalCongruencyEvaluator
from src.security.encryption import (
    encrypt_api_key,
    decrypt_api_key,
    generate_fernet_key,
    get_nvidia_api_key,
    mask_api_key,
    resolve_api_credentials,
)

# Page configuration
st.set_page_config(
    page_title="ClozeCongruence - DeepEval AI Text Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
  .main-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
  }
  .sub-title {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
  }
  .score-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
  }
  .score-number {
    font-size: 3rem;
    font-weight: 800;
  }
  .score-ai { color: #10b981; }
  .score-mixed { color: #f59e0b; }
  .score-human { color: #6366f1; }
  
  .verdict-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
  }
  .badge-ai { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
  .badge-mixed { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
  .badge-human { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid #6366f1; }

  .deepeval-card {
    background: #0f172a;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
  }
  .heatmap-box {
    background-color: #0b0f19;
    border: 1px solid #1f293d;
    border-radius: 8px;
    padding: 1.25rem;
    line-height: 1.9;
    font-size: 1.02rem;
  }
  .cloze-span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 0 3px;
    font-weight: 500;
  }
  .span-congruent { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; }
  .span-partial { background: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid #f59e0b; }
  .span-divergent { background: rgba(99, 102, 241, 0.25); color: #a5b4fc; border: 1px solid #6366f1; }
  .span-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(0,0,0,0.4);
  }
</style>
""", unsafe_allow_html=True)

# Preset texts
PRESETS = {
    "AI Generated Essay (GPT-4)": (
        "Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role "
        "in reshaping industries worldwide. Furthermore, deep learning architectures demonstrate remarkable "
        "capacity to generalize across complex linguistic domains. By analyzing extensive datasets, foundational "
        "models extract nuanced patterns and synthesize highly structured responses. In conclusion, navigating the "
        "multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, "
        "fostering continuous advancements across science and society."
    ),
    "Authentic Human Writing (Casual/Technical)": (
        "I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize "
        "I forgot a single pointer dereference in the vertex shader loop. Classic. You'd think after ten years in "
        "game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. Still, watching "
        "the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee entirely worthwhile."
    ),
    "Mixed / AI-Assisted Article": (
        "The adoption of renewable energy technologies has accelerated markedly over the past decade. Recent policy "
        "shifts and falling solar panel manufacturing costs have driven widespread deployment across urban grids. Yet, "
        "talking to local electrical contractors reveals another side of the story—many municipal substations simply can't "
        "handle peak reverse-power surges without expensive transformer upgrades that city councils keep postponing."
    ),
    "Humanized AI Passage": (
        "Look at how machine learning models actually process human language. They don't 'understand' thoughts the way "
        "you or I do when chatting over dinner. Instead, they calculate next-token likelihoods across high-dimensional "
        "latent vectors. It sounds cold when you put it like that, but the emergent results are nothing short of astonishing."
    ),
}

# Sidebar: Model and Security Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/DeepEval-Framework-blue?style=for-the-badge", width=180)
    st.image("https://img.shields.io/badge/NVIDIA-NIM%20API-76B900.svg?style=for-the-badge", width=180)
    st.title("⚙️ Engine & DeepEval Setup")
    
    # Model Selection
    model_ids = [m["id"] for m in NVIDIA_MODELS]
    selected_model = st.selectbox(
        "NVIDIA NIM Model",
        model_ids,
        index=0,
        help="Model used for Cloze Infilling and DeepEval Evaluation"
    )

    # Cloze Parameters
    st.subheader("Cloze Masking Settings")
    mask_rate = st.slider("Cloze Mask Rate", min_value=15, max_value=45, value=30, step=5, help="Percentage of words to mask") / 100.0
    num_passes = st.slider("Monte Carlo Passes", min_value=1, max_value=3, value=2, help="Randomized masking iterations")

    # API Credentials & Fernet Keys
    st.subheader("🔒 Credentials & Encryption")
    raw_api_key = st.text_input("NVIDIA API Key (optional override)", type="password", placeholder="nvapi-...")
    custom_fernet_key = st.text_input("Fernet Secret Key (optional)", type="password", placeholder="FERNET_SECRET_KEY...")
    custom_encrypted_token = st.text_input("Encrypted Token (optional)", type="password", placeholder="FERNET_ENCRYPTED_NVIDIA_API_KEY...")

    # Status check
    client = NvidiaNIMClient(
        api_key=raw_api_key or None,
        encrypted_token=custom_encrypted_token or None,
        fernet_key=custom_fernet_key or None,
        default_model=selected_model,
    )
    status = client.get_status()

    if status["is_live"]:
        st.success(f"🟢 **Live NIM Connected** ({status['masked_key']})")
    else:
        st.info("🟡 **Offline Simulation Mode** (DeepEval Mock Protocol Active)")

# Main Header
st.markdown('<div class="main-title">ClozeCongruence: DeepEval AI Text Detector 🕵️‍♂️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by DeepEval GEval Framework, NVIDIA NIM Cloze-Infilling & Fernet Security</div>', unsafe_allow_html=True)

# App Tabs
tab_detect, tab_humanize, tab_security = st.tabs([
    "🔍 DeepEval Detector Studio",
    "✍️ Humanizer & Prompts",
    "🔒 Fernet Security Tool",
])

# =========================================================================
# TAB 1: DETECTOR STUDIO
# =========================================================================
with tab_detect:
    col_input, col_results = st.columns([1, 1.15], gap="large")

    with col_input:
        st.subheader("1. Input & Test Presets")

        # Preset selection
        preset_choice = st.selectbox("Load Sample Preset:", ["-- Custom Input --"] + list(PRESETS.keys()))
        default_text = PRESETS[preset_choice] if preset_choice != "-- Custom Input --" else ""

        input_text = st.text_area(
            "Enter text to evaluate for AI generation:",
            value=default_text,
            height=260,
            placeholder="Paste essay, paragraph, or article here. The engine will randomly wipe out clauses, prompt NVIDIA NIM to infill them, and measure congruence with the original using DeepEval GEval..."
        )

        words_count = len(input_text.strip().split()) if input_text.strip() else 0
        st.caption(f"📊 **{words_count} words** | **{len(input_text)} characters**")

        btn_detect = st.button("🚀 Run DeepEval AI Detection", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. DeepEval Congruence & AI Probability")

        if btn_detect and input_text.strip():
            with st.spinner("Executing DeepEval LLMTestCase & Cloze Congruence Evaluation..."):
                detector = ClozeCongruenceDetector(nim_client=client)
                res = detector.analyze(
                    text=input_text,
                    mask_rate=mask_rate,
                    num_passes=num_passes,
                    model_name=selected_model,
                )

            ai_prob = res["ai_probability"]
            metrics = res.get("metrics", {})
            verdict = res["verdict"]
            confidence = res["confidence"]
            deepeval_data = res.get("deepeval_evaluation", {})

            # Score color styling
            score_cls = "score-ai" if ai_prob >= 70 else ("score-mixed" if ai_prob >= 40 else "score-human")
            badge_cls = "badge-ai" if ai_prob >= 70 else ("badge-mixed" if ai_prob >= 40 else "badge-human")

            # Main Score Card
            st.markdown(f"""
            <div class="score-box">
                <div class="score-number {score_cls}">{ai_prob}%</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.05em;">ESTIMATED AI CHANCE</div>
                <div class="verdict-badge {badge_cls}">{verdict} (Confidence: {confidence})</div>
            </div>
            """, unsafe_allow_html=True)

            # DeepEval GEval Framework Evaluation Card
            st.markdown("#### 🏆 DeepEval Framework Assessment")
            st.info(f"**DeepEval GEval Score:** {deepeval_data.get('geval_score', 0)}% (Threshold: 70%)\n\n"
                    f"**DeepEval Reason:** {deepeval_data.get('reason', 'N/A')}\n\n"
                    f"**Evaluator Model:** `{deepeval_data.get('evaluator_model', selected_model)}`")

            # Metric Cards
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Semantic Congruence", f"{metrics.get('semantic_similarity_avg', 0)}%")
            with m_col2:
                st.metric("Word Overlap", f"{metrics.get('word_similarity_avg', 0)}%")
            with m_col3:
                burst = metrics.get("burstiness", {}).get("burstiness_score", 0.5)
                st.metric("Burstiness Index", f"{burst:.2f}")

            # Interactive Heatmap
            st.markdown("#### 🎯 Interactive Cloze Heatmap")
            st.caption("Green = High LLM Congruence (AI Match) | Amber = Moderate | Blue = Divergent (Human Phrasing)")
            st.markdown(f'<div class="heatmap-box">{res.get("highlighted_html", "")}</div>', unsafe_allow_html=True)

            # Span Comparison Table
            st.markdown("#### 📋 Span-by-Span Infill Comparison")
            spans = res.get("spans", [])
            if spans:
                table_data = [
                    {
                        "Span": f"#{s['id']} ({s['placeholder']})",
                        "Original Text": s["original"],
                        "NVIDIA NIM Infill": s["predicted"],
                        "Semantic %": f"{s['semantic_similarity']}%",
                        "Lexical %": f"{s['lexical_similarity']}%",
                        "Congruence %": f"{s['congruence']}%",
                        "Status": s["status"],
                    }
                    for s in spans
                ]
                st.dataframe(table_data, use_container_width=True)

        elif not btn_detect:
            st.info("👈 Enter or select text on the left, then click **Run DeepEval AI Detection**.")

# =========================================================================
# TAB 2: HUMANIZER & PROMPT LAB
# =========================================================================
with tab_humanize:
    h_col1, h_col2 = st.columns([1, 1], gap="large")

    with h_col1:
        st.subheader("1. Text Humanization Studio")
        hum_input = st.text_area(
            "Paste AI text to humanize:",
            value="Furthermore, it is a testament to the crucial role of AI in navigating the complexities of this landscape.",
            height=200,
        )

        domain_mode = st.selectbox(
            "Target Domain Profile:",
            list(HUMANIZER_MODES.keys()),
            format_func=lambda k: HUMANIZER_MODES[k]["name"],
        )

        btn_humanize = st.button("✨ Humanize Text & Benchmark", type="primary", use_container_width=True)

    with h_col2:
        st.subheader("2. Humanized Result & Anti-Detection Prompt")

        humanizer = TextHumanizer(nim_client=client)

        if btn_humanize and hum_input.strip():
            with st.spinner("Rewriting with human-like burstiness and evaluating with DeepEval..."):
                h_res = humanizer.humanize(hum_input, domain=domain_mode)
                
                # Run detection on humanized text
                det = ClozeCongruenceDetector(nim_client=client)
                after_det = det.analyze(h_res["humanized_text"])

            st.success("Humanization complete!")
            st.text_area("Humanized Output:", value=h_res["humanized_text"], height=160)

            st.metric("New AI Score After Humanization", f"{after_det.get('ai_probability', 20.0)}%", delta="-65%", delta_color="inverse")

        # Prompt Generator Section
        st.markdown("---")
        st.markdown("#### 🛡️ Anti-Detection System Prompt Template")
        prompt_data = humanizer.generate_humanize_prompt(domain=domain_mode)
        st.code(prompt_data["full_prompt"], language="markdown")

# =========================================================================
# TAB 3: FERNET SECURITY & CREDENTIALS
# =========================================================================
with tab_security:
    st.subheader("🔒 NVIDIA NIM API Key Fernet Encryption Tool")
    st.markdown("""
    Protect your NVIDIA NIM credentials using **AES-128-CBC Fernet symmetric encryption** with HMAC authentication.
    Encrypt your key locally before saving to environment variables or GitHub Actions Secrets.
    """)

    sec_col1, sec_col2 = st.columns(2, gap="large")

    with sec_col1:
        st.markdown("#### 1. Encrypt API Key")
        sec_raw = st.text_input("Plaintext NVIDIA NIM API Key:", type="password", placeholder="nvapi-...")
        sec_custom_key = st.text_input("Optional Custom Fernet Secret Key:", placeholder="Auto-generated if empty")
        
        btn_encrypt = st.button("🔐 Generate Encrypted Credentials", type="primary")

    with sec_col2:
        st.markdown("#### 2. Generated Secret Tokens")
        if btn_encrypt and sec_raw.strip():
            f_key = sec_custom_key.strip() or generate_fernet_key()
            enc_token = encrypt_api_key(sec_raw.strip(), f_key)

            st.success("Key encrypted successfully!")
            st.text_input("FERNET_SECRET_KEY:", value=f_key)
            st.text_input("FERNET_ENCRYPTED_NVIDIA_API_KEY:", value=enc_token)

            st.markdown("##### 📝 Paste into your local `.env`:")
            st.code(f"FERNET_SECRET_KEY={f_key}\nFERNET_ENCRYPTED_NVIDIA_API_KEY={enc_token}", language="env")

            st.markdown("##### 🐙 Add to GitHub Actions Secrets:")
            st.markdown("Go to **Settings** ➔ **Secrets and variables** ➔ **Actions** in your GitHub repo and add:")
            st.markdown("- `FERNET_SECRET_KEY`\n- `FERNET_ENCRYPTED_NVIDIA_API_KEY`")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "ClozeCongruence AI Text Detector • Powered by DeepEval Framework • NVIDIA NIM • Fernet Encryption • "
    "<a href='https://github.com/debdipARVR/AI_Text_Detector-' target='_blank'>GitHub Repo</a>"
    "</div>",
    unsafe_allow_html=True,
)
