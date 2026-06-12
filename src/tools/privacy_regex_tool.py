import re

from langchain_core.tools import tool


@tool
def regex_privacy_tool(raw_text: str) -> dict:
    """
    Basic privacy regex tool.
    Masks clear emails and Saudi mobile numbers only.
    Extracts long numbers for LLM review.
    """
    text = str(raw_text).strip()

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text).strip()

    removed_sensitive_items = []

    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    if re.search(email_pattern, text):
        text = re.sub(email_pattern, "[EMAIL]", text)
        removed_sensitive_items.append("EMAIL")

    phone_pattern = r"(?<!\d)(?:\+966|00966|966|0)?5(?:[\s\-]?\d){8}(?!\d)"
    if re.search(phone_pattern, text):
        text = re.sub(phone_pattern, "[PHONE]", text)
        removed_sensitive_items.append("PHONE")

    long_number_candidates = []
    for match in re.finditer(r"\b\d{6,}\b", text):
        start = max(match.start() - 40, 0)
        end = min(match.end() + 40, len(text))

        long_number_candidates.append({
            "number": match.group(),
            "context": text[start:end],
        })

    return {
        "masked_text": text,
        "removed_sensitive_items": removed_sensitive_items,
        "long_number_candidates": long_number_candidates,
    }