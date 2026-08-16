"""Humanizer Engine and Prompt Generator.

Provides tools to analyze AI text markers, generate anti-detection humanizer prompts,
and rewrite text with organic human-like syntactic burstiness, varied vocabulary,
and idiosyncratic cadence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from .nim_client import NvidiaNIMClient


HUMANIZER_MODES = {
    "academic": {
        "name": "Academic & Scholarly",
        "description": "Retains intellectual rigor while introducing authorial voice, varied sentence pacing, and field-specific nuance.",
        "style_guidance": "Use active scholarly verbs, vary paragraph cadence, weave in critical questioning, and eliminate cookie-cutter transition formulas.",
    },
    "conversational": {
        "name": "Conversational & Blog",
        "description": "Engaging, direct, and voice-driven with conversational rhythm, rhetorical questions, and punchy cadence.",
        "style_guidance": "Adopt an authentic first-person or second-person perspective, use occasional contractions, interjections, and varied line lengths.",
    },
    "technical": {
        "name": "Technical & Analytical",
        "description": "Precise, pragmatic engineering voice with concrete examples and non-templated flow.",
        "style_guidance": "Focus on real-world constraints, trade-offs, direct cause-and-effect explanations, and avoid boilerplate intros.",
    },
    "creative": {
        "name": "Creative & Narrative",
        "description": "Rich sensory detail, metaphoric resonance, and highly dynamic sentence structure.",
        "style_guidance": "Emphasize show-don't-tell, asymmetric paragraph lengths, vivid verbs, and unpredictable clause sequencing.",
    },
    "business": {
        "name": "Business & Executive",
        "description": "Strategic, crisp, and persuasive communication without buzzword fatigue.",
        "style_guidance": "Lead with the bottom line, use punchy impact statements, and cut filler corporate jargon.",
    },
}

# Common cliché AI transition words & clichés
AI_CLICHE_PATTERNS = [
    (r'\bdelve into\b', 'explore / look closely at'),
    (r'\ba testament to\b', 'clear proof of'),
    (r'\bcrucial role\b', 'vital part / key factor'),
    (r'\bfoster\b', 'build / encourage'),
    (r'\blandscape\b', 'space / domain / arena'),
    (r'\bmoreover\b', 'also / on top of that'),
    (r'\bfurthermore\b', 'in addition / besides'),
    (r'\bin conclusion\b', 'ultimately / to sum up'),
    (r'\bit is worth noting that\b', 'noticeably / keep in mind that'),
    (r'\bit is essential to\b', 'we must / it pays to'),
    (r'\bnavigating the complexities\b', 'handling the nuances'),
    (r'\bmultifaceted\b', 'complex / varied'),
    (r'\bpivotal\b', 'key / deciding'),
    (r'\bnuanced\b', 'subtle / layered'),
]


class TextHumanizer:
    """Engine for generating humanizer prompts and rewriting AI text."""

    def __init__(self, nim_client: Optional[NvidiaNIMClient] = None):
        self.client = nim_client or NvidiaNIMClient()

    def generate_humanize_prompt(
        self,
        domain: str = "academic",
        target_audience: str = "General Audience",
        additional_notes: str = "",
    ) -> Dict[str, Any]:
        """Generate a battle-tested humanizer system and user prompt for LLMs."""
        mode_info = HUMANIZER_MODES.get(domain.lower(), HUMANIZER_MODES["academic"])
        
        system_prompt = (
            "You are an elite linguistic stylist and human writing reconstructor. "
            "Your objective is to rewrite text so that it reads with authentic human cadence, "
            "high structural burstiness, idiosyncratic vocabulary, and dynamic rhythm—completely "
            "free of predictable AI generation patterns and cloze predictability."
        )

        core_instructions = f"""### CORE LINGUISTIC HUMANIZATION DIRECTIVES:
1. **High Burstiness & Pacing Variance**:
   - Deliberately mix sentence lengths: follow long, compound sentences (25-35 words) with short, punchy fragments or sentences (3-7 words).
   - Avoid monotonous subject-verb-object rhythms.

2. **Eradicate AI Clichés & Stereotypical Transitions**:
   - STRICTLY FORBIDDEN words and phrases: "delve", "testament to", "crucial role", "foster", "landscape", "moreover", "furthermore", "in conclusion", "it is important to remember", "multifaceted", "pivotal".
   - Replace generic transition markers with organic topical transitions or rhetorical links.

3. **Domain Profile ({mode_info['name']})**:
   - {mode_info['style_guidance']}

4. **Preserve Core Meaning & Facts**:
   - Retain every single fact, number, thesis point, and key argument without loss of substance.

5. **Audience & Context**:
   - Target Audience: {target_audience}
   {f"- Special Focus: {additional_notes}" if additional_notes else ""}
"""

        user_prompt_template = (
            f"{core_instructions}\n"
            "### INPUT TEXT TO HUMANIZE:\n"
            "[PASTE YOUR AI OR DRAFT TEXT HERE]\n\n"
            "### OUTPUT:\n"
            "Provide ONLY the rewritten humanized text, without meta-commentary or introduction."
        )

        return {
            "domain": domain,
            "mode_name": mode_info["name"],
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "full_prompt": f"{system_prompt}\n\n{user_prompt_template}",
        }

    def analyze_ai_markers(self, text: str) -> Dict[str, Any]:
        """Scan text for common AI clichés, repetitiveness, and predictability flags."""
        found_cliches = []
        for pattern, suggestion in AI_CLICHE_PATTERNS:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                found_cliches.append({
                    "phrase": matches[0].group(0),
                    "count": len(matches),
                    "suggestion": suggestion,
                })

        # Check sentence length consistency
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
        sentences = [s.strip() for s in sentence_endings.split(text.strip()) if s.strip()]
        lengths = [len(s.split()) for s in sentences] if sentences else [0]
        
        avg_len = sum(lengths) / max(1, len(lengths))
        is_monotonous = len(lengths) >= 3 and max(lengths) - min(lengths) <= 6

        return {
            "cliche_count": len(found_cliches),
            "cliches_detected": found_cliches,
            "sentence_count": len(sentences),
            "average_sentence_length": round(avg_len, 1),
            "is_monotonous_cadence": is_monotonous,
        }

    def humanize(
        self,
        text: str,
        domain: str = "academic",
        model_name: Optional[str] = None,
        temperature: float = 0.75,
    ) -> Dict[str, Any]:
        """Rewrite input text with humanized style and high burstiness."""
        if not text or not text.strip():
            return {"humanized_text": "", "mode": domain, "ai_markers_before": {}}

        prompt_bundle = self.generate_humanize_prompt(domain=domain)
        ai_markers_before = self.analyze_ai_markers(text)

        # If live NIM client available, run generation
        if self.client.is_live and self.client.client:
            try:
                system_prompt = prompt_bundle["system_prompt"]
                user_msg = (
                    f"{prompt_bundle['user_prompt_template'].split('[PASTE YOUR AI OR DRAFT TEXT HERE]')[0]}\n"
                    f"### INPUT TEXT TO HUMANIZE:\n{text.strip()}\n\n"
                    "### OUTPUT:\nProvide ONLY the rewritten humanized text."
                )

                model = model_name or self.client.default_model
                response = self.client.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=1200,
                )
                humanized = (response.choices[0].message.content or "").strip()
                # Clean any wrapping quotes or backticks
                if humanized.startswith("```") and humanized.endswith("```"):
                    humanized = re.sub(r'^```[a-zA-Z]*\n', '', humanized)
                    humanized = re.sub(r'\n```$', '', humanized)
                
                ai_markers_after = self.analyze_ai_markers(humanized)
                return {
                    "original_text": text,
                    "humanized_text": humanized,
                    "mode": domain,
                    "is_live_generation": True,
                    "ai_markers_before": ai_markers_before,
                    "ai_markers_after": ai_markers_after,
                }
            except Exception:
                pass

        # Offline / Fallback heuristic humanizer
        humanized = self._heuristic_humanize(text)
        ai_markers_after = self.analyze_ai_markers(humanized)

        return {
            "original_text": text,
            "humanized_text": humanized,
            "mode": domain,
            "is_live_generation": False,
            "ai_markers_before": ai_markers_before,
            "ai_markers_after": ai_markers_after,
        }

    def _heuristic_humanize(self, text: str) -> str:
        """Apply rule-based stylistic transformation when offline."""
        result = text
        # 1. Replace AI clichés
        for pattern, suggestion in AI_CLICHE_PATTERNS:
            sub_choice = suggestion.split(" / ")[0]
            result = re.sub(pattern, sub_choice, result, flags=re.IGNORECASE)

        # 2. Add subtle conversational touches and contractions if appropriate
        result = re.sub(r'\bdo not\b', "don't", result)
        result = re.sub(r'\bcannot\b', "can't", result)
        result = re.sub(r'\bit is\b', "it's", result)

        return result
