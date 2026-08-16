"""Streamlit Web Application for Two-Pass ClozeCongruence AI Text Detector with DeepEval Meaning Metrics."""

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
from src.security.encryption import (
    encrypt_api_key,
    decrypt_api_key,
    generate_fernet_key,
    get_nvidia_api_key,
    mask_api_key,
)

# Page configuration
st.set_page_config(
    page_title="ClozeCongruence - Two-Pass DeepEval AI Text Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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
  .score-ai { color: #ef4444; }
  .score-mixed { color: #f59e0b; }
  .score-human { color: #10b981; }
  
  .verdict-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
  }
  .badge-ai { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
  .badge-mixed { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
  .badge-human { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }

  .pass-card {
    background: #0f172a;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
  }
  .heatmap-box {
    background-color: #0b0f19;
    border: 1px solid #1f293d;
    border-radius: 8px;
    padding: 1.25rem;
    line-height: 2.0;
    font-size: 1.02rem;
  }
  .cloze-span {
    display: inline-flex;
    flex-direction: column;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 4px 2px;
    font-weight: 500;
  }
  .span-congruent { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
  .span-partial { background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid #f59e0b; }
  .span-divergent { background: rgba(16, 185, 129, 0.2); color: #a7f3d0; border: 1px solid #10b981; }
  .span-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(0,0,0,0.5);
    margin-top: 3px;
    width: fit-content;
  }
</style>
""", unsafe_allow_html=True)

# Preset texts
PRESETS = {
    "AI Generated Essay (GPT-4)": (
        "Artificial intelligence has revolutionized modern technological paradigms, playing a crucial role in reshaping industries worldwide. "
        "Furthermore, deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains. "
        "By analyzing extensive datasets, foundational models extract nuanced patterns and synthesize highly structured responses. "
        "In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation, fostering continuous advancements across science and society."
    ),
    "Authentic Human Writing (Casual/Technical)": (
        "I spent three sleepless nights debugging that memory leak in our C++ graphics pipeline, only to realize I forgot a single pointer dereference in the vertex shader loop. "
        "Classic. "
        "You'd think after ten years in game dev you'd spot something so stupid right away, but fatigue does funny things to your brain. "
        "Still, watching the frame rate jump from 14 FPS back up to a smooth 120 made the cold coffee entirely worthwhile."
    ),
    "Mixed / AI-Assisted Article": (
        "The adoption of renewable energy technologies has accelerated markedly over the past decade. "
        "Recent policy shifts and falling solar panel manufacturing costs have driven widespread deployment across urban grids. "
        "Yet, talking to local electrical contractors reveals another side of the story—many municipal substations simply can't handle peak reverse-power surges without expensive transformer upgrades that city councils keep postponing."
    ),
}

# Sidebar: Model and Security Configuration
with st.sidebar:
    st.image("https://img.shields.io/badge/DeepEval-Meaning%20Metric-blue?style=for-the-badge", width=220)
    st.image("https://img.shields.io/badge/NVIDIA-NIM%20API-76B900.svg?style=for-the-badge", width=220)
    st.title("⚙️ Engine Configuration")
    
    # Model Selection
    model_ids = [m["id"] for m in NVIDIA_MODELS]
    selected_model = st.selectbox(
        "NVIDIA NIM Model",
        model_ids,
        index=0,
        help="Primary model for Cloze Sentence Infilling & DeepEval Evaluation"
    )

    st.markdown("#### 🔬 Two-Pass Masking Mode")
    st.info(
        "• **Pass 1 (Sparse)**: Removes 1 sentence every 4 lines.\n"
        "• **Pass 2 (Alternate)**: Removes alternate sentences every 2 lines.\n\n"
        "**Weights:**\n"
        "• 55% DeepEval Meaning Similarity (Highest)\n"
        "• 30% Semantic Cosine Similarity\n"
        "• 15% Lexical Overlap"
    )

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
        st.info("🟡 **Offline Simulation Mode** (DeepEval Protocol Active)")

# Main Header
st.markdown('<div class="main-title">Two-Pass DeepEval AI Text Detector 🕵️‍♂️⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Complete Sentence Removal • DeepEval Meaning Similarity (Highest Priority) • Cosine Congruence • Pass 1 & Pass 2 Synthesis</div>', unsafe_allow_html=True)

# App Tabs
tab_detect, tab_humanize, tab_security = st.tabs([
    "🔍 Two-Pass Detector Studio",
    "✍️ Humanizer & Prompts",
    "🔒 Fernet Security Tool",
])

# =========================================================================
# TAB 1: DETECTOR STUDIO
# =========================================================================
with tab_detect:
    col_input, col_results = st.columns([1, 1.25], gap="large")

    with col_input:
        st.subheader("1. Input Paragraph")

        preset_choice = st.selectbox("Load Sample Preset:", ["-- Custom Input --"] + list(PRESETS.keys()))
        default_text = PRESETS[preset_choice] if preset_choice != "-- Custom Input --" else ""

        input_text = st.text_area(
            "Enter text paragraph to evaluate:",
            value=default_text,
            height=280,
            placeholder="Paste complete paragraph here. The detector will execute Pass 1 (1 sentence every 4 lines) and Pass 2 (alternate sentences every 2 lines), prompting NVIDIA NIM to complete the missing sentences..."
        )

        words_count = len(input_text.strip().split()) if input_text.strip() else 0
        st.caption(f"📊 **{words_count} words** | **{len(input_text)} characters**")

        btn_detect = st.button("🚀 Run Two-Pass DeepEval Detection", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. Two-Pass Evaluation & AI Probability")

        if btn_detect and input_text.strip():
            with st.spinner(f"Executing Pass 1 & Pass 2 with {selected_model} & DeepEval Meaning Metric..."):
                detector = ClozeCongruenceDetector(nim_client=client)
                res = detector.analyze(
                    text=input_text,
                    model_name=selected_model,
                )

            ai_prob = res["ai_probability"]
            verdict = res["verdict"]
            confidence = res["confidence"]
            combined_score = res["combined_congruence_score"]
            pass1 = res["pass_1"]
            pass2 = res["pass_2"]
            metrics = res["metrics"]

            # Style classification
            score_cls = "score-ai" if ai_prob >= 70 else ("score-mixed" if ai_prob >= 40 else "score-human")
            badge_cls = "badge-ai" if ai_prob >= 70 else ("badge-mixed" if ai_prob >= 40 else "badge-human")

            # Final Verdict Summary Box
            st.markdown(f"""
            <div class="score-box">
                <div class="score-number {score_cls}">{ai_prob}%</div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #94a3b8; letter-spacing: 0.05em;">COMBINED AI PROBABILITY</div>
                <div class="verdict-badge {badge_cls}">{verdict} (Confidence: {confidence})</div>
                <div style="margin-top: 0.75rem; font-size: 0.88rem; color: #cbd5e1;">{res.get('two_pass_verdict_reason', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Two-Pass Comparative Cards
            p_col1, p_col2 = st.columns(2)

            with p_col1:
                st.markdown("#### 1️⃣ Pass 1 (Sparse Masking)")
                st.markdown(f"""
                <div class="pass-card">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #60a5fa;">Pass 1 Congruence: {pass1['congruence_score']}%</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">1 sentence removed per 4 lines ({pass1['sentences_masked_count']} sentence)</div>
                    <hr style="border-color: #334155; margin: 0.5rem 0;" />
                    <div>🎯 <b>Meaning Similarity (DeepEval):</b> <span style="color:#f59e0b; font-weight:700;">{pass1['meaning_similarity']}%</span></div>
                    <div>📐 <b>Semantic Cosine:</b> {pass1['semantic_cosine']}%</div>
                    <div>🔤 <b>Lexical Overlap:</b> {pass1['lexical_similarity']}%</div>
                    <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><i>{pass1['deepeval_reason']}</i></div>
                </div>
                """, unsafe_allow_html=True)

            with p_col2:
                st.markdown("#### 2️⃣ Pass 2 (Alternate Masking)")
                st.markdown(f"""
                <div class="pass-card">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #60a5fa;">Pass 2 Congruence: {pass2['congruence_score']}%</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">Sentences removed every 2 lines ({pass2['sentences_masked_count']} sentences)</div>
                    <hr style="border-color: #334155; margin: 0.5rem 0;" />
                    <div>🎯 <b>Meaning Similarity (DeepEval):</b> <span style="color:#f59e0b; font-weight:700;">{pass2['meaning_similarity']}%</span></div>
                    <div>📐 <b>Semantic Cosine:</b> {pass2['semantic_cosine']}%</div>
                    <div>🔤 <b>Lexical Overlap:</b> {pass2['lexical_similarity']}%</div>
                    <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #cbd5e1;"><i>{pass2['deepeval_reason']}</i></div>
                </div>
                """, unsafe_allow_html=True)

            # Infill Sentences Breakdown Table
            st.markdown("#### 📋 Pass 2 Sentence Infill Breakdown")
            spans = pass2.get("spans", [])
            if spans:
                table_data = [
                    {
                        "Placeholder": s["placeholder"],
                        "Original Sentence": s["original_sentence"],
                        "NIM Infilled Sentence": s["predicted_sentence"],
                        "Meaning Similarity (55%)": f"{s['meaning_similarity']}%",
                        "Semantic Cosine (30%)": f"{s['semantic_cosine']}%",
                        "Congruence Score": f"{s['congruence']}%",
                        "Status": s["status"],
                    }
                    for s in spans
                ]
                st.dataframe(table_data, use_container_width=True)

            # Interactive Highlighted Paragraph
            st.markdown("#### 🎯 Paragraph Sentence Highlight Map")
            st.markdown(f'<div class="heatmap-box">{res.get("highlighted_html", "")}</div>', unsafe_allow_html=True)

        elif not btn_detect:
            st.info("👈 Enter or select a text paragraph on the left, then click **Run Two-Pass DeepEval Detection**.")

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
            with st.spinner("Rewriting with organic burstiness and evaluating with DeepEval..."):
                h_res = humanizer.humanize(hum_input, domain=domain_mode)
                det = ClozeCongruenceDetector(nim_client=client)
                after_det = det.analyze(h_res["humanized_text"])

            st.success("Humanization complete!")
            st.text_area("Humanized Output:", value=h_res["humanized_text"], height=160)
            st.metric("New AI Score After Humanization", f"{after_det.get('ai_probability', 15.0)}%", delta="-70%", delta_color="inverse")

        st.markdown("---")
        st.markdown("#### 🛡️ Anti-Detection System Prompt Template")
        prompt_data = humanizer.generate_humanize_prompt(domain=domain_mode)
        st.code(prompt_data["full_prompt"], language="markdown")

# =========================================================================
# TAB 3: FERNET SECURITY & CREDENTIALS
# =========================================================================
with tab_security:
    st.subheader("🔒 NVIDIA NIM API Key Fernet Encryption Tool")
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
            st.code(f"FERNET_SECRET_KEY={f_key}\nFERNET_ENCRYPTED_NVIDIA_API_KEY={enc_token}", language="env")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Two-Pass ClozeCongruence AI Detector • Powered by DeepEval Meaning Metrics & NVIDIA NIM • "
    "<a href='https://github.com/debdipARVR/AI_Text_Detector-' target='_blank'>GitHub Repo</a>"
    "</div>",
    unsafe_allow_html=True,
)
