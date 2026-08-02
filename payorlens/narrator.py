"""
narrator.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("payorlens.narrator")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
NARRATOR_ENABLED = os.getenv("PAYORLENS_NARRATOR_ENABLED", "true").lower() == "true"

SYSTEM_PROMPT = """You are writing a short plain-English risk summary for a health \
insurance compliance officer who is not a data scientist. You will be given a JSON \
object of model evaluation metrics (data quality, performance, fairness, robustness, \
and an overall risk verdict). Rules:

- Use ONLY the numbers given to you. Never estimate, round differently, calculate \
  new figures, or invent any statistic not present in the input.
- Do not mention or infer anything about individual patients or claims — you were \
  not given any patient-level data.
- Write 3-5 sentences: what the single biggest risk finding is, why it matters for \
  a deployment decision, and one concrete recommended next step.
- Respond with ONLY a JSON object matching exactly:
  {"summary": "...", "top_risk": "...", "recommended_action": "..."}
"""


def _numbers_in(text: str) -> set[str]:
    """Every numeric token in a string, e.g. '12', '0.34', '85.0'."""
    return set(re.findall(r"\d+\.?\d*", text))


def _fallback_narrative(metrics: dict[str, Any]) -> dict[str, str]:
    verdict = metrics.get("overall_risk") or "UNKNOWN"
    critical = metrics.get("critical_count")
    high = metrics.get("high_count")
    recommendation = metrics.get("recommendation") or "Review the full governance report."

    counts = []
    if critical is not None:
        counts.append(f"{critical} critical finding(s)")
    if high is not None:
        counts.append(f"{high} high-severity finding(s)")
    counts_str = " and ".join(counts) if counts else "no severity breakdown available"

    return {
        "summary": (
            f"Overall risk verdict for this run: {verdict} ({counts_str}). "
            f"{recommendation}"
        ),
        "top_risk": f"Overall risk verdict: {verdict}",
        "recommended_action": recommendation,
    }


def generate_narrative(metrics: dict[str, Any]) -> dict[str, str]:
    """Returns {"summary", "top_risk", "recommended_action"}."""
    if not NARRATOR_ENABLED or not GEMINI_API_KEY:
        logger.info("Narrator disabled or no GEMINI_API_KEY set — using fallback narrative.")
        return _fallback_narrative(metrics)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("`google-genai` package not installed — using fallback narrative.")
        return _fallback_narrative(metrics)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=json.dumps(metrics, default=str),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        
        raw_text = response.text.strip()
        parsed = json.loads(raw_text)
    except Exception:
        logger.exception("Narrator LLM call failed — using fallback narrative.")
        return _fallback_narrative(metrics)

    required_keys = {"summary", "top_risk", "recommended_action"}
    if not required_keys.issubset(parsed.keys()):
        logger.warning("Narrator response missing required keys — using fallback narrative.")
        return _fallback_narrative(metrics)


    output_numbers = _numbers_in(" ".join(parsed[k] for k in required_keys))
    allowed_numbers = _numbers_in(json.dumps(metrics, default=str))
    hallucinated = output_numbers - allowed_numbers

    if hallucinated:
        logger.warning(
            "Narrator invented number(s) not present in metrics (%s) — using fallback narrative.",
            hallucinated,
        )
        return _fallback_narrative(metrics)

    return {k: parsed[k] for k in required_keys}