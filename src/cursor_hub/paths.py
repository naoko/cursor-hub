"""OS-specific path resolution and JSON/JSONC helpers."""

import json
import os
import platform
from pathlib import Path
from typing import Optional


def cursor_settings_path() -> Optional[Path]:
    """Return the Cursor settings.json path for this OS, or None."""
    system = platform.system()
    if system == "Darwin":
        p = Path.home() / "Library/Application Support/Cursor/User/settings.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        p = Path(appdata) / "Cursor/User/settings.json"
    else:  # Linux / WSL
        p = Path.home() / ".config/Cursor/User/settings.json"
    return p if p.exists() else None


def cursor_workspace_storage() -> Optional[Path]:
    """Return the Cursor workspaceStorage directory."""
    system = platform.system()
    if system == "Darwin":
        p = Path.home() / "Library/Application Support/Cursor/User/workspaceStorage"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        p = Path(appdata) / "Cursor/User/workspaceStorage"
    else:
        p = Path.home() / ".config/Cursor/User/workspaceStorage"
    return p if p.exists() else None


def state_db_path() -> Path:
    """Return the path to Cursor's global state.vscdb."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library/Application Support/Cursor/User/globalStorage/state.vscdb"
        )
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Cursor/User/globalStorage/state.vscdb"
    else:
        return Path.home() / ".config/Cursor/User/globalStorage/state.vscdb"


def _strip_jsonc_comments(text: str) -> str:
    """Strip JS-style comments from JSONC text without corrupting string values."""
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            result.append(text[i:j])
            i = j
        elif text[i : i + 2] == "//":
            while i < n and text[i] != "\n":
                i += 1
        elif text[i : i + 2] == "/*":
            end = text.find("*/", i + 2)
            i = end + 2 if end != -1 else n
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def read_json(path: Path) -> dict:
    """Read a JSON or JSONC file, returning {} on any failure."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_strip_jsonc_comments(text))
    except Exception:
        return {}
