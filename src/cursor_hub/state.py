"""Composer session state, model detection, and context estimation."""

import base64
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from .models import context_window_for
from .paths import (
    cursor_settings_path,
    cursor_workspace_storage,
    read_json,
    state_db_path,
)

# Keys Cursor uses in settings.json for the active model (in priority order)
_MODEL_KEYS = [
    "cursor.chat.defaultModel",
    "cursor.general.inlineEditModel",
    "cursor.cpp.defaultModel",
    "cursor.general.aiModel",
]


def get_model_from_settings(settings: dict) -> str:
    """Extract the active model name from a Cursor settings dict."""
    for key in _MODEL_KEYS:
        val = settings.get(key, "")
        if val:
            return val
    nested = settings.get("cursor", {}).get("chat", {}).get("defaultModel", "")
    if nested:
        return nested
    return ""


@dataclass
class ComposerState:
    """State extracted from Cursor's Composer database."""

    name: str = ""
    model: str = ""
    unified_mode: str = ""
    force_mode: str = ""
    is_agentic: bool = False
    max_mode: bool = False
    context_used: int = 0
    context_limit: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    files_edited: list[str] = field(default_factory=list)
    subagents: list[dict[str, str]] = field(default_factory=list)
    project: str = ""
    composer_id: str = ""
    pending_question_count: int = 0
    awaiting_user_input: bool = False


def _extract_state_from_data(data: dict) -> ComposerState:
    """Extract a ComposerState from a composerData JSON dict."""
    state = ComposerState()
    state.name = data.get("name", "")
    state.unified_mode = data.get("unifiedMode", "")
    state.force_mode = data.get("forceMode", "")
    state.is_agentic = bool(data.get("isAgentic", False))
    state.context_used = data.get("contextTokensUsed", 0) or 0
    state.context_limit = data.get("contextTokenLimit", 0) or 0

    model_config = data.get("modelConfig", {})
    if isinstance(model_config, dict):
        model_name = model_config.get("modelName", "")
        if model_name and isinstance(model_name, str):
            state.model = model_name
            state.max_mode = bool(model_config.get("maxMode", False))
    if not state.model:
        for fld in ("modelType", "selectedModelId", "model", "modelId"):
            m = data.get(fld, "")
            if m and isinstance(m, str):
                state.model = m
                break
    return state


def _extract_tool_usage(
    data: dict,
    cur: sqlite3.Cursor,
) -> tuple[dict[str, int], list[str], list[dict[str, str]], int]:
    """
    Extract tool usage from a composerData entry's conversation history.

    The conversationState is a base64-encoded protobuf containing 32-byte
    SHA256 hashes (prefixed by 0x0a 0x20) that point to agentKv:blob:<hex>
    entries in the same database.

    Returns (tool_counts, files_edited, subagents, pending_question_count).
    """
    cs = data.get("conversationState", "")
    if not cs:
        return {}, [], [], 0

    try:
        raw = base64.b64decode(cs.lstrip("~"))
    except Exception:
        return {}, [], [], 0

    # Extract 32-byte hashes from protobuf (field tag 0x0a, length 0x20)
    hashes: list[str] = []
    i = 0
    while i < len(raw) - 33:
        if raw[i] == 0x0A and raw[i + 1] == 0x20:
            hashes.append(raw[i + 2 : i + 34].hex())
            i += 34
        else:
            i += 1

    if not hashes:
        return {}, [], [], 0

    tool_counts: dict[str, int] = {}
    files_edited: list[str] = []
    seen_files: set[str] = set()
    subagents: list[dict[str, str]] = []
    ask_calls: set[str] = set()
    ask_results: set[str] = set()

    for h in hashes:
        cur.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"agentKv:blob:{h}",),
        )
        blob_row = cur.fetchone()
        if not blob_row or not blob_row[0]:
            continue
        try:
            text = (
                blob_row[0]
                if isinstance(blob_row[0], str)
                else blob_row[0].decode("utf-8", errors="replace")
            )
            blob = json.loads(text)
            for c in blob.get("content", []):
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "tool-call":
                    tool_name = c.get("toolName", c.get("name", ""))
                    if not tool_name:
                        continue
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                    args = c.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    # Track file edits
                    if tool_name in ("Write", "StrReplace", "search_replace"):
                        p = args.get("path", args.get("filePath", ""))
                        if p:
                            short = p.rsplit("/", 1)[-1]
                            if short not in seen_files:
                                seen_files.add(short)
                                files_edited.append(short)
                    # Track subagent spawns
                    if tool_name == "Task" and args.get("description"):
                        subagents.append(
                            {
                                "type": args.get("subagent_type", "default"),
                                "description": args.get("description", ""),
                            }
                        )
                    if tool_name == "AskQuestion":
                        call_id = c.get("toolCallId", "")
                        if call_id:
                            ask_calls.add(call_id)
                elif ctype == "tool-result":
                    if c.get("toolName") == "AskQuestion":
                        call_id = c.get("toolCallId", "")
                        if call_id:
                            ask_results.add(call_id)
        except (json.JSONDecodeError, TypeError):
            continue

    pending_question_count = len(ask_calls - ask_results)
    return tool_counts, files_edited, subagents, pending_question_count


def _pending_questions_from_bubbles(cur: sqlite3.Cursor, composer_id: str) -> int:
    """
    Infer pending structured questions from bubble records.

    Cursor's question UI can be persisted as ask_question toolFormer bubbles even when
    conversationState blobs don't have a matching unresolved AskQuestion tool-call.
    """
    try:
        cur.execute(
            "SELECT value FROM cursorDiskKV WHERE key LIKE ?",
            (f"bubbleId:{composer_id}:%",),
        )
        rows = cur.fetchall()
    except Exception:
        return 0

    ask_times: list[str] = []
    assistant_reply_times: list[str] = []
    for (value_str,) in rows:
        try:
            data = json.loads(value_str)
        except (json.JSONDecodeError, TypeError):
            continue
        created = str(data.get("createdAt", ""))
        if not created:
            continue
        tool_former = data.get("toolFormerData", {})
        if isinstance(tool_former, dict) and tool_former.get("name") == "ask_question":
            ask_times.append(created)
        # When a structured question is answered, Cursor emits a regular assistant
        # message bubble (type=2 with text) after the modal response.
        if data.get("type") == 2 and data.get("text"):
            assistant_reply_times.append(created)

    if not ask_times:
        return 0

    pending = 0
    for ask_ts in ask_times:
        if not any(reply_ts > ask_ts for reply_ts in assistant_reply_times):
            pending += 1
    return pending


def _find_workspace_folder(project_dir: Path) -> Optional[Path]:
    """Find the Cursor workspaceStorage folder for the given project."""
    ws_storage = cursor_workspace_storage()
    if not ws_storage:
        return None

    def normalize_path(value: str) -> str:
        if not value:
            return ""
        parsed = urlparse(value)
        candidate = unquote(parsed.path) if parsed.scheme == "file" else value
        return str(Path(candidate).resolve(strict=False))

    try:
        target = str(project_dir.resolve(strict=False))
        for folder in ws_storage.iterdir():
            ws_json = folder / "workspace.json"
            if not ws_json.exists():
                continue
            data = read_json(ws_json)
            folder_path = normalize_path(str(data.get("folder", "")))
            if folder_path == target:
                return folder
    except Exception:
        pass
    return None


def get_active_sessions(project_dir: Path) -> list[ComposerState]:
    """
    Return the currently open Composer sessions for a project.

    Reads selectedComposerIds from the workspace-level state.vscdb, then
    fetches full session data from the global state.vscdb.
    """
    ws_folder = _find_workspace_folder(project_dir)
    if not ws_folder:
        return []

    ws_db = ws_folder / "state.vscdb"
    if not ws_db.exists():
        return []

    try:
        with sqlite3.connect(f"file:{ws_db}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
            )
            row = cur.fetchone()

        if not row or not row[0]:
            return []

        ws_data = json.loads(row[0])
        selected_ids = ws_data.get("selectedComposerIds", [])
        if not selected_ids:
            return []

        all_composers = {c["composerId"]: c for c in ws_data.get("allComposers", [])}

        global_db = state_db_path()
        if not global_db.exists():
            return []

        sessions = []
        with sqlite3.connect(f"file:{global_db}?mode=ro", uri=True) as conn2:
            cur2 = conn2.cursor()
            for cid in selected_ids:
                cur2.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?",
                    (f"composerData:{cid}",),
                )
                row = cur2.fetchone()
                if row and row[0]:
                    try:
                        data = json.loads(row[0])
                        state = _extract_state_from_data(data)
                        state.composer_id = cid
                        tc, fe, sa, pending_count = _extract_tool_usage(data, cur2)
                        pending_count = max(
                            pending_count, _pending_questions_from_bubbles(cur2, cid)
                        )
                        state.tool_counts = tc
                        state.files_edited = fe
                        state.subagents = sa
                        state.pending_question_count = pending_count
                        state.awaiting_user_input = pending_count > 0
                        sessions.append(state)
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    summary = all_composers.get(cid, {})
                    if summary:
                        state = ComposerState(
                            name=summary.get("name", ""),
                            unified_mode=summary.get("unifiedMode", ""),
                            force_mode=summary.get("forceMode", ""),
                            composer_id=cid,
                        )
                        sessions.append(state)

        return sessions
    except Exception:
        return []


def get_all_active_sessions() -> dict[str, list[ComposerState]]:
    """
    Return active sessions across ALL Cursor workspaces.

    Returns a dict mapping project name to its list of open sessions.
    """
    ws_storage = cursor_workspace_storage()
    if not ws_storage:
        return {}

    global_db = state_db_path()
    if not global_db.exists():
        return {}

    result: dict[str, list[ComposerState]] = {}

    try:
        global_conn = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True)
        global_cur = global_conn.cursor()

        for folder in ws_storage.iterdir():
            ws_json = folder / "workspace.json"
            ws_db = folder / "state.vscdb"
            if not ws_json.exists() or not ws_db.exists():
                continue

            try:
                ws_data_raw = read_json(ws_json)
                folder_uri = ws_data_raw.get("folder", "")
                project_name = folder_uri.rstrip("/").rsplit("/", 1)[-1]
                if not project_name:
                    continue

                with sqlite3.connect(f"file:{ws_db}?mode=ro", uri=True) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT value FROM ItemTable "
                        "WHERE key = 'composer.composerData'"
                    )
                    row = cur.fetchone()

                if not row or not row[0]:
                    continue

                composer_data = json.loads(row[0])
                selected_ids = composer_data.get("selectedComposerIds", [])
                if not selected_ids:
                    continue

                all_composers = {
                    c["composerId"]: c for c in composer_data.get("allComposers", [])
                }

                sessions: list[ComposerState] = []
                for cid in selected_ids:
                    global_cur.execute(
                        "SELECT value FROM cursorDiskKV WHERE key = ?",
                        (f"composerData:{cid}",),
                    )
                    row = global_cur.fetchone()
                    if row and row[0]:
                        try:
                            data = json.loads(row[0])
                            state = _extract_state_from_data(data)
                            state.composer_id = cid
                            tc, fe, sa, pending_count = _extract_tool_usage(
                                data, global_cur
                            )
                            pending_count = max(
                                pending_count,
                                _pending_questions_from_bubbles(global_cur, cid),
                            )
                            state.tool_counts = tc
                            state.files_edited = fe
                            state.subagents = sa
                            state.pending_question_count = pending_count
                            state.awaiting_user_input = pending_count > 0
                            state.project = project_name
                            sessions.append(state)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    else:
                        summary = all_composers.get(cid, {})
                        if summary:
                            state = ComposerState(
                                name=summary.get("name", ""),
                                unified_mode=summary.get("unifiedMode", ""),
                                force_mode=summary.get("forceMode", ""),
                                project=project_name,
                                composer_id=cid,
                            )
                            sessions.append(state)

                if sessions:
                    result[project_name] = sessions
            except Exception:
                continue

        global_conn.close()
    except Exception:
        pass

    return result


def get_composer_state(project_dir: Optional[Path] = None) -> ComposerState:
    """
    Read Composer state from Cursor's databases.

    If project_dir is given, returns the most recently updated open session
    for that project. Otherwise falls back to the most recent global entry.
    """
    if project_dir:
        sessions = get_active_sessions(project_dir)
        if sessions:
            return sessions[0]

    db_path = state_db_path()
    if not db_path.exists():
        return ComposerState()

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
            )
            best_ts = -1
            best_data = None
            for (value_str,) in cur.fetchall():
                try:
                    data = json.loads(value_str)
                    ts = data.get("lastUpdatedAt") or data.get("createdAt") or 0
                    if ts > best_ts:
                        best_ts = ts
                        best_data = data
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue

        if best_data:
            return _extract_state_from_data(best_data)
        return ComposerState()
    except Exception:
        return ComposerState()


def get_model_from_state_db() -> str:
    """Read the active model from Cursor's state database."""
    return get_composer_state().model


def get_model(project_dir: Path) -> str:
    """
    Model resolution order:
    1. CURSOR_MODEL env var (user override)
    2. Project .cursor/settings.json
    3. Global Cursor settings.json
    4. Cursor state.vscdb (Composer 1.5+)
    5. 'unknown'
    """
    env_model = os.environ.get("CURSOR_MODEL", "").strip()
    if env_model:
        return env_model

    proj_settings = read_json(project_dir / ".cursor" / "settings.json")
    m = get_model_from_settings(proj_settings)
    if m:
        return m

    sp = cursor_settings_path()
    if sp:
        m = get_model_from_settings(read_json(sp))
        if m:
            return m

    m = get_model_from_state_db()
    if m:
        return m

    return "unknown"


def estimate_context_tokens(project_dir: Path, model_str: str) -> tuple[int, int]:
    """
    Estimate tokens in the current Cursor composer/chat context.

    Returns (used_tokens, max_tokens).  Token estimate: ~4 chars ≈ 1 token.
    """
    max_tokens = context_window_for(model_str)

    target_folder = _find_workspace_folder(project_dir)
    if not target_folder:
        return 0, max_tokens

    used_chars = 0
    counted_paths: set[Path] = set()
    for fname in ("cursor-chat-data", "composer", "composer.json", "chat.json"):
        candidate = target_folder / fname
        if candidate.exists() and candidate.is_file():
            try:
                used_chars += candidate.stat().st_size
                counted_paths.add(candidate.resolve())
            except Exception:
                pass

    try:
        for p in target_folder.glob("*.json"):
            if p.name in ("workspace.json", "settings.json"):
                continue
            try:
                rp = p.resolve()
                if rp in counted_paths:
                    continue
                used_chars += p.stat().st_size
            except Exception:
                pass
    except Exception:
        pass

    used_tokens = used_chars // 4
    return min(used_tokens, max_tokens), max_tokens
