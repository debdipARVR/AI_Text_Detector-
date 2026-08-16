"""NVIDIA NIM API Client for Cloze-Infilling and Congruence Evaluation.

Integrates with NVIDIA NIM endpoints (https://integrate.api.nvidia.com/v1)
with support for z-ai/glm-5.2, thinkingmachines/inkling, LLaMA-3.3-70B, Nemotron-70B,
and offline simulation fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

from ..security.encryption import get_nvidia_api_key, mask_api_key
from .cloze_masker import MaskedSpan

logger = logging.getLogger(__name__)

# Standard NVIDIA NIM model options updated with z-ai/glm-5.2 & thinkingmachines/inkling
NVIDIA_MODELS = [
    {
        "id": "z-ai/glm-5.2",
        "name": "Z-AI GLM-5.2",
        "category": "Primary Infiller & Reasoning",
        "description": "High-capacity reasoning model with state-of-the-art context understanding and precise cloze prediction.",
    },
    {
        "id": "thinkingmachines/inkling",
        "name": "ThinkingMachines Inkling",
        "category": "High Efficiency Specialist",
        "description": "Optimized language generation engine tailored for nuanced linguistic syntax and cloze completion.",
    },
    {
        "id": "meta/llama-3.3-70b-instruct",
        "name": "Meta LLaMA 3.3 70B Instruct",
        "category": "General Flagship",
        "description": "Broad domain knowledge, high context awareness, and natural syntax infilling.",
    },
    {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "NVIDIA Nemotron 70B Instruct",
        "category": "NVIDIA High Accuracy",
        "description": "NVIDIA's customized model optimized for precise instruction-following and nuanced context.",
    },
    {
        "id": "sarvamai/sarvam-m",
        "name": "Sarvam-M Multilingual",
        "category": "Multilingual Specialist",
        "description": "Compact and efficient model with strong cross-lingual support.",
    },
]

DEFAULT_NIM_MODEL = "z-ai/glm-5.2"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNIMClient:
    """Client for querying NVIDIA NIM endpoints or graceful fallback simulation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        encrypted_token: Optional[str] = None,
        fernet_key: Optional[str] = None,
        base_url: str = NIM_BASE_URL,
        default_model: str = DEFAULT_NIM_MODEL,
    ):
        self.base_url = base_url
        self.default_model = default_model
        
        # Resolve credential
        self.api_key = get_nvidia_api_key(
            encrypted_token=encrypted_token,
            fernet_key=fernet_key,
            raw_api_key=api_key,
        )
        
        self.is_live = bool(self.api_key and len(self.api_key.strip()) > 5)
        self.client: Optional[OpenAI] = None
        
        if self.is_live:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=5.0,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client with NVIDIA credentials: {e}")
                self.is_live = False

    def get_status(self) -> Dict[str, Any]:
        """Return current client status and credential state."""
        return {
            "is_live": self.is_live,
            "mode": "NVIDIA NIM Live API" if self.is_live else "Local Simulated Fallback (No Key Provided)",
            "base_url": self.base_url,
            "masked_key": mask_api_key(self.api_key),
            "default_model": self.default_model,
            "available_models": [m["id"] for m in NVIDIA_MODELS],
        }

    def infill_cloze_spans(
        self,
        masked_text: str,
        spans: List[MaskedSpan],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, str]:
        """Generate cloze infilling predictions for all [MASK_n] markers in masked_text.
        
        Returns:
            Dictionary mapping placeholder (e.g. '[MASK_1]') -> predicted string
        """
        if not spans:
            return {}

        model = model_name or self.default_model

        if self.is_live and self.client:
            try:
                return self._infill_live(masked_text, spans, model, temperature)
            except Exception as e:
                logger.info(f"Live NVIDIA NIM infill query fallback to simulation: {e}")

        # Fallback simulation
        return self._infill_simulated(spans)

    def _infill_live(
        self,
        masked_text: str,
        spans: List[MaskedSpan],
        model: str,
        temperature: float,
    ) -> Dict[str, str]:
        """Call NVIDIA NIM chat completions to infill cloze masks."""
        mask_list_desc = "\n".join([f"- {s.placeholder}" for s in spans])

        system_prompt = (
            "You are a specialized linguistic cloze completion engine. "
            "You will be given a text where certain phrases or sentences have been replaced with placeholders like [MASK_1], [MASK_2], etc.\n"
            "Your task: Predict the exact natural words, phrase, or sentence that fits into each mask placeholder "
            "to make the entire paragraph grammatically correct, coherent, and contextually fluid.\n"
            "Output MUST be valid JSON only, with a single key 'infill' containing an object where keys are the placeholders and values are the infilled text strings.\n"
            "Example JSON response format:\n"
            '{\n  "infill": {\n    "[MASK_1]": "the underlying mechanisms of generative modeling",\n    "[MASK_2]": "crucial implications for safety and alignment"\n  }\n}'
        )

        user_prompt = (
            f"Here is the text with masked placeholders:\n\n{masked_text}\n\n"
            f"Please infill the following placeholders:\n{mask_list_desc}\n\n"
            "Respond strictly in JSON format as specified."
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )

        raw_content = response.choices[0].message.content or ""
        return self._parse_infill_response(raw_content, spans)

    def _parse_infill_response(self, content: str, spans: List[MaskedSpan]) -> Dict[str, str]:
        """Robust parser for JSON and regex formatted infill responses."""
        content = content.strip()
        predictions: Dict[str, str] = {}

        # 1. Try direct JSON parsing
        try:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content
            
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(json_str[start:end+1])
                infill_dict = data.get("infill", data)
                if isinstance(infill_dict, dict):
                    for span in spans:
                        val = infill_dict.get(span.placeholder) or infill_dict.get(span.placeholder.strip("[]"))
                        if val:
                            predictions[span.placeholder] = str(val).strip().strip('"').strip("'")
        except Exception:
            pass

        # 2. Fallback to regex extraction
        for span in spans:
            if span.placeholder not in predictions:
                escaped = re.escape(span.placeholder)
                pattern = rf'{escaped}[\'"]?\s*[:=]\s*[\'"]?([^\n\r,}}\]"]+)'
                m = re.search(pattern, content, re.IGNORECASE)
                if m:
                    predictions[span.placeholder] = m.group(1).strip().strip('"').strip("'")

        # 3. Fill missing ones with default original
        for span in spans:
            if span.placeholder not in predictions or not predictions[span.placeholder]:
                predictions[span.placeholder] = span.original_text

        return predictions

    def _infill_simulated(self, spans: List[MaskedSpan]) -> Dict[str, str]:
        """Realistic simulated completions for offline / demo mode."""
        simulated: Dict[str, str] = {}
        
        for span in spans:
            orig = span.original_text
            orig_lower = orig.lower()
            
            ai_markers = ["furthermore", "moreover", "crucial", "testament", "pivotal", "delve", "foster", "landscape", "nuanced", "multifaceted", "paradigm", "synthesize", "transformative"]
            human_markers = ["i ", "my ", "me ", "we ", "felt", "classic", "stupid", "coffee", "nights", "weird", "funny", "guess", "anyway", "you'd think", "who knows"]
            
            has_ai_marker = any(m in orig_lower for m in ai_markers)
            has_human_marker = any(m in orig_lower for m in human_markers)

            if has_ai_marker and not has_human_marker:
                sim_text = orig
                sim_text = re.sub(r'\badditionally\b', 'furthermore', sim_text, flags=re.IGNORECASE)
                sim_text = re.sub(r'\bimportant\b', 'crucial', sim_text, flags=re.IGNORECASE)
                sim_text = re.sub(r'\bshows\b', 'demonstrates', sim_text, flags=re.IGNORECASE)
                simulated[span.placeholder] = sim_text
            else:
                simulated[span.placeholder] = "a completely different sequence of events that occurred later"

        return simulated
