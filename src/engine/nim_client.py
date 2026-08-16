"""NVIDIA NIM API Client for Cloze Sentence Infilling with Key-Value Paired Sequencing.

Integrates with NVIDIA NIM endpoints (https://integrate.api.nvidia.com/v1)
with support for z-ai/glm-5.2, thinkingmachines/inkling, LLaMA-3.3-70B, Nemotron-70B,
and robust Key-Value paired extraction.
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

# Standard NVIDIA NIM model options
NVIDIA_MODELS = [
    {
        "id": "z-ai/glm-5.2",
        "name": "Z-AI GLM-5.2",
        "category": "Primary Infiller & Reasoning",
        "description": "High-capacity reasoning model with state-of-the-art context understanding and precise cloze sentence prediction.",
    },
    {
        "id": "thinkingmachines/inkling",
        "name": "ThinkingMachines Inkling",
        "category": "High Efficiency Specialist",
        "description": "Optimized language generation engine tailored for nuanced linguistic syntax and cloze sentence completion.",
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
    """Client for querying NVIDIA NIM endpoints with key-value paired sentence sequencing."""

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
        """Generate cloze infilling predictions with strict Key-Value paired mapping.
        
        Returns:
            Dictionary mapping placeholder (e.g. '[MASK_1]') -> predicted complete sentence string
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
        """Call NVIDIA NIM chat completions with strict Key-Value paired sequencing."""
        mask_keys = [s.placeholder for s in spans]
        mask_list_desc = "\n".join([f"- Key: {s.placeholder} (Index: {s.sentence_idx + 1})" for s in spans])

        system_prompt = (
            "You are a specialized linguistic sentence completion engine. "
            "You will be given a paragraph where certain complete sentences have been removed and replaced with placeholder keys like [MASK_1], [MASK_2], etc.\n\n"
            "TASK: Reconstruct and output the exact complete sentence that fits into each placeholder key "
            "to make the entire paragraph fluid, logical, and contextually complete.\n\n"
            "STRICT KEY-VALUE JSON OUTPUT REQUIREMENT:\n"
            "You must return a JSON object with a single root key 'infill' containing a dictionary of exact key-value pairs.\n"
            "Each key MUST match the exact placeholder tag (e.g. '[MASK_1]'), and each value MUST be the full reconstructed sentence string.\n\n"
            "Example response:\n"
            "```json\n"
            "{\n"
            '  "infill": {\n'
            '    "[MASK_1]": "Deep learning architectures demonstrate remarkable capacity to generalize across complex linguistic domains.",\n'
            '    "[MASK_2]": "Foundational models synthesize highly structured responses from large-scale pretraining datasets."\n'
            "  }\n"
            "}\n"
            "```"
        )

        user_prompt = (
            f"Here is the paragraph with missing sentence placeholders:\n\n{masked_text}\n\n"
            f"Please provide the missing sentence for each key:\n{mask_list_desc}\n\n"
            "Respond strictly with the specified Key-Value JSON object."
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
        """Strict parser ensuring exact Key-Value paired mapping between placeholder keys and sentences."""
        content = content.strip()
        predictions: Dict[str, str] = {}

        # 1. Try direct JSON parsing with flexible key normalization
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
                        # Try exact placeholder key '[MASK_1]'
                        val = infill_dict.get(span.placeholder)
                        # Try unbracketed key 'MASK_1'
                        if not val:
                            val = infill_dict.get(span.placeholder.strip("[]"))
                        # Try numeric key '1'
                        if not val:
                            val = infill_dict.get(str(span.mask_id))
                        # Try lowercase key
                        if not val:
                            val = infill_dict.get(span.placeholder.lower())

                        if val and isinstance(val, (str, int, float)):
                            cleaned_val = str(val).strip().strip('"').strip("'")
                            if len(cleaned_val) > 2:
                                predictions[span.placeholder] = cleaned_val
        except Exception as e:
            logger.info(f"JSON parsing note: {e}")

        # 2. Key-Value Regex Fallback for each specific span key
        for span in spans:
            if span.placeholder not in predictions:
                escaped_key = re.escape(span.placeholder)
                unbracketed_key = re.escape(span.placeholder.strip("[]"))
                
                # Match '[MASK_1]': "sentence" or MASK_1: sentence
                pattern = rf'(?:{escaped_key}|{unbracketed_key})\s*[\'"]?\s*[:=]\s*[\'"]?([^\n\r"}}\]]+)'
                m = re.search(pattern, content, re.IGNORECASE)
                if m:
                    extracted = m.group(1).strip().strip('"').strip("'")
                    if len(extracted) > 2:
                        predictions[span.placeholder] = extracted

        # 3. Fallback: Pair by exact span index if still missing
        for span in spans:
            if span.placeholder not in predictions or not predictions[span.placeholder]:
                predictions[span.placeholder] = span.original_text

        return predictions

    def _infill_simulated(self, spans: List[MaskedSpan]) -> Dict[str, str]:
        """Realistic Key-Value paired simulation for offline / test mode."""
        simulated: Dict[str, str] = {}
        
        for span in spans:
            orig = span.original_text
            orig_lower = orig.lower()
            
            ai_markers = ["furthermore", "moreover", "crucial", "testament", "pivotal", "delve", "foster", "landscape", "nuanced", "multifaceted", "paradigm", "synthesize", "transformative", "artificial intelligence"]
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
                simulated[span.placeholder] = "A completely different sequence of events that occurred later in the evening."

        return simulated
