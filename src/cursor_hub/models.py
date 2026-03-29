"""Model display classification and fallback context windows."""

import re

# Small provider-family style map: keeps UX consistent without giant per-model tables.
_PROVIDER_STYLES: list[tuple[str, str, str, str]] = [
    ("claude", "◈", "Claude", "magenta"),
    ("gpt", "⬢", "GPT", "bright_green"),
    ("composer", "⊙", "Composer", "bright_cyan"),
    ("o1", "⬡", "o1", "bright_cyan"),
    ("o3", "⬡", "o3", "bright_cyan"),
    ("o4", "⬡", "o4", "bright_cyan"),
    ("gemini", "✦", "Gemini", "yellow"),
    ("grok", "✗", "Grok", "bright_red"),
    ("deepseek", "⬡", "DeepSeek", "bright_blue"),
    ("cursor", "⊙", "Cursor", "bright_white"),
    ("default", "⊙", "Default", "bright_white"),
]


def _family_style(model_str: str) -> tuple[str, str, str]:
    """Return (icon, family_label, color) for model identifier."""
    lower = model_str.lower()
    for fragment, icon, label, color in _PROVIDER_STYLES:
        if fragment in lower:
            return icon, label, color
    return "◉", "Model", "bright_white"


def _normalize_model_suffix(model_str: str) -> str:
    """Trim common suffix noise from raw model IDs."""
    cleaned = model_str.strip()
    for suffix in (
        "-high-thinking",
        "-thinking",
        "-latest",
        "-preview",
        "-default",
    ):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    cleaned = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", cleaned)
    cleaned = re.sub(r"-\d{8}$", "", cleaned)
    return cleaned


def classify_model(model_str: str) -> tuple[str, str]:
    """Return (display_label, rich_color) for a model identifier string."""
    if not model_str or model_str == "unknown":
        return "? No model set", "dim white"

    normalized = _normalize_model_suffix(model_str)
    icon, family_label, color = _family_style(normalized)
    lower = normalized.lower()

    if lower.startswith("claude-"):
        detail = normalized[len("claude-") :].replace("-", " ").title()
        if "opus" in lower:
            color = "bright_magenta"
            icon = "◆"
        elif "haiku" in lower:
            color = "bright_blue"
            icon = "◇"
        return f"{icon} {family_label} {detail}", color
    if lower.startswith("gpt-"):
        detail = normalized[len("gpt-") :].upper().replace("4O", "4o")
        return f"{icon} {family_label}-{detail}", color
    if lower.startswith("composer-"):
        detail = normalized[len("composer-") :].replace("-", " ")
        return f"{icon} {family_label} {detail}", color
    if re.match(r"^o\d", lower):
        return f"{icon} {normalized}", color
    if lower.startswith("gemini-"):
        detail = normalized[len("gemini-") :].replace("-", " ").title()
        return f"{icon} {family_label} {detail}", color
    if lower.startswith("cursor-"):
        detail = normalized[len("cursor-") :].replace("-", " ").title()
        return f"{icon} {family_label} {detail}", color
    if lower in ("default", "cursor-default"):
        return f"{icon} {family_label}", color

    return f"{icon} {normalized}", color


def context_window_for(model_str: str) -> int:
    """
    Return fallback context window for a model string.

    Note: active Composer sessions should use live contextTokenLimit from Cursor state.
    """
    lower = model_str.lower()
    if "claude" in lower:
        return 200_000
    if "gpt-4.1" in lower:
        return 1_047_576
    if "gpt-4" in lower or "gpt-3.5" in lower:
        return 128_000
    if re.search(r"\bo[134]\b", lower):
        return 200_000
    if "gemini-1.5-pro" in lower:
        return 2_000_000
    if "gemini-2.0-flash" in lower:
        return 1_048_576
    if "gemini" in lower:
        return 1_000_000
    if "grok" in lower:
        return 131_072
    if "deepseek" in lower:
        return 128_000
    return 128_000
