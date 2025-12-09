# src/utils.py
import re

def clean_text(text: str) -> str:
    """
    Basic text preprocessing: lowercase + remove special characters
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()
