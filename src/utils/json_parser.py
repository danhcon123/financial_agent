"""
Robust JSON extraction from LLM outputs.

Handles common failure modes:
1. Markdown code fences (```json...```)
2. Text before/after JSON
3. Malformed JSON with recoverable structure
"""

import json
import re
from typing import Any, Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_json_safely(
    text: str,
    fallback_key: str = "thesis",
    default_value: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract JSON from LLM output with robust error handling.

    Args:
        text: Raw LLM response text
        fallback_key: Key to use if wrapping raw text
        default_value: Return this if all parsing fails (prevent crashes)

    Returns:
        Parsed JSON dict or safe fallback
    """
    # Handle already parsed JSON (from LangChain)
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        logger.warning("Received list instead of dict, wrapping in default structure")
        return default_value or {}

    # Original string handling
    if not text or not text.strip() or not isinstance(text, str):
        logger.warning("Empty text provided to JSON parser")
        return default_value or {}
    
    text = text.strip()

    # Step 1: Remove markdown code fences
    if text.startswith("```"):
        text = _strip_code_fences(text)

    # Step 2: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: Heuristic extraction - find first { to last }
    json_obj = _extract_json_object(text)
    if json_obj:
        return json_obj
    
    # Step 4: Fallback - wrap raw text
    logger.warning(f"Could not parse JSON, wrapping in fallback key '{fallback_key}'")
    if default_value is not None:
        return default_value
    return {fallback_key: text[:500]} # Truncate to prevent huge objects


def _strip_code_fences(text: str) -> str:
    """Remove ```json...``` or ```json...``` fences"""
    text = text.strip("`").strip()

    if "\n" in text:
        # Drop optional leading "json" or "JSON"
        first_line, rest = text.split("\n", 1)
        if first_line.strip().lower() in ("json", ""):
            return rest.strip()
        
    return text

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Heuristic: extract content between first { and last }""" 
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None
    
    candidate = text[first_brace : last_brace + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def safe_list_extract(data: Dict[str, Any], key: str, max_items: int = 10) -> list[str]:
    """
    Safely extract and validate list of strings from parsed JSON.

    Args:
        data: Parsed JSON dict
        key: Key to extract
        max_items: Maximum list length (prevent huge outputs)
    Returns:
        List of non-empty strings
    """
    raw = data.get(key, [])

    if not isinstance(raw, list):
        logger.warning(f"Expected list for key '{key}', got {type(raw)}")
        return []
    
    # Filter to non-empty strings and truncate
    result = []
    for item in raw:
        text = str(item).strip()
        if text:
            result.append(text)

    if len(result) > max_items:
        logger.warning(f"Truncating list '{key}' from {len(result)} to {max_items} items")
        result = result[:max_items]

    return result