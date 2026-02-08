"""File I/O helpers with safe error handling"""
import json
import os
from typing import Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

def write_json(path: str, payload: Any) -> None:
    """Write JSON to file, creating parent directories"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    logger.debug(f"Wrote JSON to {path}")
    
def safe_write_json(path: str, payload: Any) -> bool:
    """
    Write JSON with exception handling.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        write_json(path, payload)
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        return False

def read_json(path: str) -> Any:
    """Read JSON from file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

        