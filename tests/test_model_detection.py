"""Tests for model detection in cursor-hub."""

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from cursor_hub.dashboard import (
    _action_todo_row,
    _CONTEXT_HISTORY,
    _eta_minutes_to_threshold,
    _format_eta_minutes,
    _format_mode,
    _format_session_row,
    _format_subagent_rows,
    _quick_action_for_context,
    _trend_delta_per_10m,
)
from cursor_hub.main import _build_json_snapshot, _positive_float
from cursor_hub.models import classify_model, context_window_for
from cursor_hub.state import (
    ComposerState,
    get_active_sessions,
    get_composer_state,
    get_model,
    get_model_from_settings,
    get_model_from_state_db,
)


# ── Shared helpers ───────────────────────────────────────────────────────────


def create_test_db(
    db_path: Path,
    composer_data: dict | None = None,
    item_data: dict | None = None,
) -> None:
    """Create a minimal state.vscdb with test data."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)"
    )
    if item_data:
        for k, v in item_data.items():
            conn.execute("INSERT INTO ItemTable (key, value) VALUES (?, ?)", (k, v))
    if composer_data:
        for k, v in composer_data.items():
            conn.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                (k, json.dumps(v)),
            )
    conn.commit()
    conn.close()


def make_global_storage_dir(td_path: Path) -> Path:
    """Create the macOS globalStorage directory tree, return path to state.vscdb."""
    gs_dir = (
        td_path
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "globalStorage"
    )
    gs_dir.mkdir(parents=True)
    return gs_dir / "state.vscdb"


@contextmanager
def mock_darwin_home(td_path: Path):
    """Mock platform.system and Path.home for macOS-style paths."""
    with mock.patch("cursor_hub.paths.platform.system", return_value="Darwin"):
        with mock.patch("cursor_hub.paths.Path.home", return_value=td_path):
            yield


@contextmanager
def without_cursor_model_env():
    """Ensure CURSOR_MODEL is not set in the environment."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CURSOR_MODEL", None)
        yield


# ── get_model_from_settings ─────────────────────────────────────────────────


class TestGetModelFromSettings:
    def test_returns_empty_when_no_keys(self):
        assert get_model_from_settings({}) == ""

    def test_reads_chat_default_model(self):
        settings = {"cursor.chat.defaultModel": "claude-sonnet-4"}
        assert get_model_from_settings(settings) == "claude-sonnet-4"

    def test_reads_inline_edit_model(self):
        settings = {"cursor.general.inlineEditModel": "gpt-4o"}
        assert get_model_from_settings(settings) == "gpt-4o"

    def test_reads_legacy_key(self):
        settings = {"cursor.cpp.defaultModel": "gpt-4"}
        assert get_model_from_settings(settings) == "gpt-4"

    def test_reads_older_key(self):
        settings = {"cursor.general.aiModel": "claude-3-5-sonnet"}
        assert get_model_from_settings(settings) == "claude-3-5-sonnet"

    def test_priority_order(self):
        settings = {
            "cursor.chat.defaultModel": "claude-opus-4",
            "cursor.general.aiModel": "gpt-4o",
        }
        assert get_model_from_settings(settings) == "claude-opus-4"

    def test_nested_object_shape(self):
        settings = {"cursor": {"chat": {"defaultModel": "gemini-2.5-pro"}}}
        assert get_model_from_settings(settings) == "gemini-2.5-pro"

    def test_skips_empty_string_values(self):
        settings = {
            "cursor.chat.defaultModel": "",
            "cursor.general.aiModel": "o3",
        }
        assert get_model_from_settings(settings) == "o3"


# ── classify_model ───────────────────────────────────────────────────────────


class TestClassifyModel:
    def test_claude_4_5_opus(self):
        label, color = classify_model("claude-4.5-opus-high-thinking")
        assert "4.5 Opus" in label
        assert color == "bright_magenta"

    def test_claude_opus(self):
        label, color = classify_model("claude-opus-4-20250514")
        assert "Opus" in label
        assert color == "bright_magenta"

    def test_claude_sonnet(self):
        label, color = classify_model("claude-sonnet-4-20250514")
        assert "Sonnet" in label
        assert color == "magenta"

    def test_gpt4o(self):
        label, color = classify_model("gpt-4o-2024-08-06")
        assert "GPT-4o" in label

    def test_unknown_model(self):
        label, color = classify_model("unknown")
        assert "No model set" in label
        assert "dim" in color

    def test_unrecognized_model_shows_raw(self):
        label, color = classify_model("some-new-model-v2")
        assert "some-new-model-v2" in label
        assert color == "bright_white"

    def test_empty_string(self):
        label, _ = classify_model("")
        assert "No model set" in label


# ── context_window_for ───────────────────────────────────────────────────────


class TestContextWindowFor:
    def test_known_model(self):
        assert context_window_for("claude-opus-4") == 200_000

    def test_gemini_large_context(self):
        assert context_window_for("gemini-1.5-pro") == 2_000_000

    def test_gpt4_1(self):
        assert context_window_for("gpt-4.1-turbo") == 1_047_576

    def test_fallback_for_unknown(self):
        assert context_window_for("some-mystery-model") == 128_000


# ── CLI arg parsing helpers ───────────────────────────────────────────────────
class TestPositiveFloat:
    def test_accepts_positive_number(self):
        assert _positive_float("2.5") == 2.5

    def test_rejects_zero(self):
        with pytest.raises(SystemExit):
            with mock.patch("sys.stderr", new=StringIO()):
                parser = argparse.ArgumentParser()
                parser.add_argument("--interval", type=_positive_float)
                parser.parse_args(["--interval", "0"])

    def test_rejects_negative(self):
        with pytest.raises(SystemExit):
            with mock.patch("sys.stderr", new=StringIO()):
                parser = argparse.ArgumentParser()
                parser.add_argument("--interval", type=_positive_float)
                parser.parse_args(["--interval", "-1"])


class TestJsonSnapshot:
    def test_snapshot_with_active_agents(self, tmp_path):
        session = ComposerState(
            name="Agent one",
            model="claude-sonnet-4",
            unified_mode="agent",
            force_mode="edit",
            context_used=1000,
            context_limit=2000,
            tool_counts={"ReadFile": 4, "ApplyPatch": 2},
            files_edited=["main.py"],
            subagents=[{"type": "explore", "description": "scan"}],
        )
        with mock.patch("cursor_hub.main.count_rules_files", return_value=2):
            with mock.patch("cursor_hub.main.count_mcps", return_value=1):
                with mock.patch("cursor_hub.main.count_notepads", return_value=3):
                    with mock.patch(
                        "cursor_hub.main.cursor_process_info",
                        return_value=(True, "123 MB"),
                    ):
                        with mock.patch(
                            "cursor_hub.main.get_active_sessions",
                            return_value=[session],
                        ):
                            payload = _build_json_snapshot(tmp_path, 0.0)
        assert payload["project"] == str(tmp_path)
        assert payload["config"]["rules"] == 2
        assert payload["active_agent_count"] == 1
        assert payload["agents"][0]["name"] == "Agent one"
        assert payload["agents"][0]["context_pct"] == 50.0
        assert payload["agents"][0]["top_tools"][0]["name"] == "ReadFile"

    def test_snapshot_without_agents_falls_back_to_model_context(self, tmp_path):
        composer = ComposerState(
            unified_mode="chat", context_used=500, context_limit=1000
        )
        with mock.patch("cursor_hub.main.count_rules_files", return_value=0):
            with mock.patch("cursor_hub.main.count_mcps", return_value=0):
                with mock.patch("cursor_hub.main.count_notepads", return_value=0):
                    with mock.patch(
                        "cursor_hub.main.cursor_process_info", return_value=(False, "—")
                    ):
                        with mock.patch(
                            "cursor_hub.main.get_active_sessions", return_value=[]
                        ):
                            with mock.patch(
                                "cursor_hub.main.get_composer_state",
                                return_value=composer,
                            ):
                                with mock.patch(
                                    "cursor_hub.main.get_model", return_value="gpt-4o"
                                ):
                                    payload = _build_json_snapshot(tmp_path, 0.0)
        assert payload["model"] == "gpt-4o"
        assert payload["mode"] == "chat"
        assert payload["context"]["pct"] == 50.0


# ── get_model (integration) ─────────────────────────────────────────────────


class TestGetModel:
    def test_env_var_takes_priority(self, tmp_path):
        with mock.patch.dict("os.environ", {"CURSOR_MODEL": "gpt-4o"}):
            assert get_model(tmp_path) == "gpt-4o"

    def test_project_settings_override(self, tmp_path):
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        settings = {"cursor.chat.defaultModel": "claude-sonnet-4"}
        (cursor_dir / "settings.json").write_text(json.dumps(settings))

        with without_cursor_model_env():
            assert get_model(tmp_path) == "claude-sonnet-4"

    def test_falls_back_to_unknown(self, tmp_path):
        with without_cursor_model_env():
            with mock.patch("cursor_hub.state.cursor_settings_path", return_value=None):
                with mock.patch(
                    "cursor_hub.state.get_model_from_state_db", return_value=""
                ):
                    assert get_model(tmp_path) == "unknown"

    def test_state_db_fallback(self, tmp_path):
        with without_cursor_model_env():
            with mock.patch("cursor_hub.state.cursor_settings_path", return_value=None):
                with mock.patch(
                    "cursor_hub.state.get_model_from_state_db",
                    return_value="claude-opus-4",
                ):
                    assert get_model(tmp_path) == "claude-opus-4"


# ── get_model_from_state_db ─────────────────────────────────────────────────


class TestGetModelFromStateDb:
    def test_returns_empty_when_no_db(self):
        with tempfile.TemporaryDirectory() as td:
            with mock_darwin_home(Path(td) / "nope"):
                assert get_model_from_state_db() == ""

    def test_reads_model_from_model_config(self):
        """Composer 1.5+ stores model in modelConfig.modelName."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)

            create_test_db(
                db_path,
                composer_data={
                    "composerData:abc-123": {
                        "modelConfig": {"modelName": "claude-4.5-opus-high-thinking"},
                        "contextTokenLimit": 176000,
                        "lastUpdatedAt": 1000,
                    }
                },
            )

            with mock_darwin_home(td_path):
                result = get_model_from_state_db()
                assert result == "claude-4.5-opus-high-thinking"

    def test_falls_back_to_flat_model_fields(self):
        """When modelConfig is absent, falls back to flat fields."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)

            create_test_db(
                db_path,
                composer_data={
                    "composerData:abc-123": {
                        "modelType": "gpt-4o",
                        "contextTokenLimit": 128000,
                        "lastUpdatedAt": 1000,
                    }
                },
            )

            with mock_darwin_home(td_path):
                result = get_model_from_state_db()
                assert result == "gpt-4o"

    def test_model_config_takes_priority_over_flat_fields(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)

            create_test_db(
                db_path,
                composer_data={
                    "composerData:abc-123": {
                        "modelConfig": {"modelName": "claude-4.5-opus-high-thinking"},
                        "modelType": "gpt-4o",
                        "lastUpdatedAt": 1000,
                    }
                },
            )

            with mock_darwin_home(td_path):
                result = get_model_from_state_db()
                assert result == "claude-4.5-opus-high-thinking"

    def test_picks_most_recently_updated_entry(self):
        """Should use lastUpdatedAt, not key order, to find the active session."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)

            create_test_db(
                db_path,
                composer_data={
                    "composerData:zzz-999": {
                        "modelConfig": {"modelName": "claude-4.5-opus-high-thinking"},
                        "lastUpdatedAt": 1000,
                    },
                    "composerData:aaa-111": {
                        "modelConfig": {"modelName": "composer-1.5"},
                        "lastUpdatedAt": 2000,
                    },
                },
            )

            with mock_darwin_home(td_path):
                result = get_model_from_state_db()
                assert result == "composer-1.5"

    def test_handles_corrupt_json_gracefully(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)

            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                ("composerData:bad", "not valid json{{{"),
            )
            conn.commit()
            conn.close()

            with mock_darwin_home(td_path):
                result = get_model_from_state_db()
                assert result == ""


# ── get_composer_state ───────────────────────────────────────────────────────


class TestGetComposerState:
    def test_extracts_mode_from_composer_data(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)
            create_test_db(
                db_path,
                composer_data={
                    "composerData:abc-123": {
                        "modelConfig": {
                            "modelName": "claude-4.5-opus-high-thinking",
                            "maxMode": False,
                        },
                        "unifiedMode": "agent",
                        "forceMode": "edit",
                        "isAgentic": True,
                        "contextTokensUsed": 30106,
                        "contextTokenLimit": 176000,
                        "lastUpdatedAt": 1000,
                    }
                },
            )
            with mock_darwin_home(td_path):
                state = get_composer_state()
                assert state.model == "claude-4.5-opus-high-thinking"
                assert state.unified_mode == "agent"
                assert state.force_mode == "edit"
                assert state.is_agentic is True
                assert state.max_mode is False
                assert state.context_used == 30106
                assert state.context_limit == 176000

    def test_agent_plan_mode(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)
            create_test_db(
                db_path,
                composer_data={
                    "composerData:xyz-789": {
                        "modelConfig": {"modelName": "claude-4.5-opus-high-thinking"},
                        "unifiedMode": "agent",
                        "forceMode": "plan",
                        "isAgentic": True,
                        "lastUpdatedAt": 1000,
                    }
                },
            )
            with mock_darwin_home(td_path):
                state = get_composer_state()
                assert state.unified_mode == "agent"
                assert state.force_mode == "plan"

    def test_returns_empty_state_when_no_db(self):
        with tempfile.TemporaryDirectory() as td:
            with mock_darwin_home(Path(td) / "nope"):
                state = get_composer_state()
                assert state.model == ""
                assert state.unified_mode == ""

    def test_max_mode(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            db_path = make_global_storage_dir(td_path)
            create_test_db(
                db_path,
                composer_data={
                    "composerData:abc": {
                        "modelConfig": {"modelName": "gpt-4o", "maxMode": True},
                        "unifiedMode": "chat",
                        "lastUpdatedAt": 1000,
                    }
                },
            )
            with mock_darwin_home(td_path):
                state = get_composer_state()
                assert state.max_mode is True
                assert state.unified_mode == "chat"


# ── _format_mode ─────────────────────────────────────────────────────────────


class TestFormatMode:
    def test_agent_edit(self):
        state = ComposerState(unified_mode="agent", force_mode="edit")
        text = _format_mode(state)
        assert "Agent" in text.plain
        assert "Edit" in text.plain

    def test_agent_plan(self):
        state = ComposerState(unified_mode="agent", force_mode="plan")
        text = _format_mode(state)
        assert "Agent" in text.plain
        assert "Plan" in text.plain

    def test_chat_only(self):
        state = ComposerState(unified_mode="chat", force_mode="chat")
        text = _format_mode(state)
        assert "Chat" in text.plain
        assert text.plain.count("Chat") == 1

    def test_max_mode_shown(self):
        state = ComposerState(unified_mode="agent", force_mode="edit", max_mode=True)
        text = _format_mode(state)
        assert "MAX" in text.plain

    def test_empty_state(self):
        state = ComposerState()
        text = _format_mode(state)
        assert text.plain.strip() == "—"


# ── get_active_sessions ──────────────────────────────────────────────────────


class TestGetActiveSessions:
    def _setup_workspace(
        self,
        td_path: Path,
        project_dir: Path,
        selected_ids: list[str],
        all_composers: list[dict],
    ):
        """Set up a workspace storage folder with composer state."""
        ws_storage = (
            td_path
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "workspaceStorage"
        )
        ws_folder = ws_storage / "test-workspace"
        ws_folder.mkdir(parents=True)

        ws_json = {"folder": f"file://{project_dir}"}
        (ws_folder / "workspace.json").write_text(json.dumps(ws_json))

        ws_db = ws_folder / "state.vscdb"
        conn = sqlite3.connect(str(ws_db))
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        composer_data = {
            "allComposers": all_composers,
            "selectedComposerIds": selected_ids,
        }
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            ("composer.composerData", json.dumps(composer_data)),
        )
        conn.commit()
        conn.close()
        return ws_folder

    def _setup_global_db(self, td_path: Path, composer_entries: dict):
        """Set up the global state.vscdb with composerData entries."""
        gs_dir = (
            td_path
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "globalStorage"
        )
        gs_dir.mkdir(parents=True, exist_ok=True)
        db_path = gs_dir / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)"
        )
        for k, v in composer_entries.items():
            conn.execute(
                "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                (k, json.dumps(v)),
            )
        conn.commit()
        conn.close()

    def test_returns_two_open_sessions(self, tmp_path):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            self._setup_workspace(
                td_path,
                tmp_path,
                selected_ids=["session-aaa", "session-bbb"],
                all_composers=[
                    {"composerId": "session-aaa", "name": "First agent"},
                    {"composerId": "session-bbb", "name": "Second agent"},
                ],
            )
            self._setup_global_db(
                td_path,
                {
                    "composerData:session-aaa": {
                        "name": "First agent",
                        "modelConfig": {"modelName": "composer-1.5"},
                        "unifiedMode": "agent",
                        "forceMode": "edit",
                        "contextTokensUsed": 10000,
                        "contextTokenLimit": 200000,
                    },
                    "composerData:session-bbb": {
                        "name": "Second agent",
                        "modelConfig": {"modelName": "claude-4.5-opus-high-thinking"},
                        "unifiedMode": "agent",
                        "forceMode": "plan",
                        "contextTokensUsed": 50000,
                        "contextTokenLimit": 200000,
                    },
                },
            )

            with mock_darwin_home(td_path):
                sessions = get_active_sessions(tmp_path)
                assert len(sessions) == 2
                assert sessions[0].name == "First agent"
                assert sessions[0].model == "composer-1.5"
                assert sessions[0].unified_mode == "agent"
                assert sessions[0].context_used == 10000
                assert sessions[1].name == "Second agent"
                assert sessions[1].model == "claude-4.5-opus-high-thinking"
                assert sessions[1].force_mode == "plan"

    def test_returns_empty_for_unknown_project(self, tmp_path):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ws_storage = (
                td_path
                / "Library"
                / "Application Support"
                / "Cursor"
                / "User"
                / "workspaceStorage"
            )
            ws_storage.mkdir(parents=True)

            with mock_darwin_home(td_path):
                sessions = get_active_sessions(tmp_path)
                assert sessions == []

    def test_falls_back_to_summary_when_global_missing(self, tmp_path):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            self._setup_workspace(
                td_path,
                tmp_path,
                selected_ids=["session-ccc"],
                all_composers=[
                    {
                        "composerId": "session-ccc",
                        "name": "Fallback session",
                        "unifiedMode": "chat",
                        "forceMode": "edit",
                    },
                ],
            )
            self._setup_global_db(td_path, {})

            with mock_darwin_home(td_path):
                sessions = get_active_sessions(tmp_path)
                assert len(sessions) == 1
                assert sessions[0].name == "Fallback session"
                assert sessions[0].unified_mode == "chat"


# ── _format_session_row ──────────────────────────────────────────────────────


class TestFormatSessionRow:
    def test_shows_model_and_mode(self):
        s = ComposerState(
            name="My agent",
            model="composer-1.5",
            unified_mode="agent",
            force_mode="edit",
            context_used=10000,
            context_limit=200000,
        )
        text = _format_session_row(s, 0, history_key="test:model-mode")
        assert "Composer 1.5" in text.plain
        assert "Agent" in text.plain
        assert "My agent" in text.plain
        assert "5%" in text.plain

    def test_shows_context_percentage(self):
        s = ComposerState(
            model="gpt-4o",
            unified_mode="chat",
            context_used=100000,
            context_limit=200000,
        )
        text = _format_session_row(s, 0, history_key="test:ctx-pct")
        assert "50%" in text.plain

    def test_max_mode_shown_once(self):
        s = ComposerState(
            model="gpt-4o",
            unified_mode="agent",
            force_mode="edit",
            max_mode=True,
            context_used=1000,
            context_limit=200000,
        )
        text = _format_session_row(s, 0, history_key="test:max-mode")
        assert text.plain.count("MAX") == 1

    def test_no_subagent_inline_when_present(self):
        """Subagents should be shown as tree rows, not inline."""
        s = ComposerState(
            model="composer-1.5",
            unified_mode="agent",
            subagents=[
                {"type": "explore", "description": "Explore codebase"},
            ],
        )
        text = _format_session_row(s, 0, history_key="test:no-subagent-inline")
        # Subagents are rendered separately via _format_subagent_rows
        assert "sub:" not in text.plain


class TestFormatSubagentRows:
    def test_tree_view_with_multiple_subagents(self):
        s = ComposerState(
            model="composer-1.5",
            unified_mode="agent",
            subagents=[
                {"type": "explore", "description": "Explore codebase"},
                {"type": "generalPurpose", "description": "Create components"},
                {"type": "browser-use", "description": "Check live site"},
            ],
        )
        rows = _format_subagent_rows(s)
        assert len(rows) == 3
        assert "explore" in rows[0].plain
        assert "Explore codebase" in rows[0].plain
        assert "generalPurpose" in rows[1].plain
        assert "browser-use" in rows[2].plain
        # First two use ├─, last uses └─
        assert "├─" in rows[0].plain
        assert "├─" in rows[1].plain
        assert "└─" in rows[2].plain

    def test_single_subagent_uses_last_connector(self):
        s = ComposerState(
            subagents=[{"type": "explore", "description": "Check structure"}],
        )
        rows = _format_subagent_rows(s)
        assert len(rows) == 1
        assert "└─" in rows[0].plain

    def test_empty_when_no_subagents(self):
        s = ComposerState()
        rows = _format_subagent_rows(s)
        assert rows == []

    def test_icons_per_type(self):
        s = ComposerState(
            subagents=[
                {"type": "explore", "description": "a"},
                {"type": "browser-use", "description": "b"},
            ]
        )
        rows = _format_subagent_rows(s)
        assert "🔍" in rows[0].plain
        assert "🌐" in rows[1].plain


class TestTrendDelta:
    def teardown_method(self):
        _CONTEXT_HISTORY.clear()

    def test_returns_none_with_insufficient_history(self):
        from collections import deque

        history = deque([(1000.0, 10.0)])
        assert _trend_delta_per_10m(history, 11.0, 1005.0) is None

    def test_projects_delta_to_10_minutes(self):
        from collections import deque

        # +2% in 5 minutes => +4%/10m projection
        history = deque([(1000.0, 40.0), (1300.0, 42.0)])
        delta = _trend_delta_per_10m(history, 42.0, 1300.0)
        assert delta == pytest.approx(4.0)


class TestQuickActionHint:
    def test_no_hint_below_threshold(self):
        assert _quick_action_for_context(79.9) is None

    def test_warns_at_80_plus(self):
        hint = _quick_action_for_context(80.0)
        assert hint is not None
        assert "Context is high" in hint.plain

    def test_urgent_at_90_plus(self):
        hint = _quick_action_for_context(90.0)
        assert hint is not None
        assert "Start a new session now" in hint.plain

    def test_hint_threshold_override_env(self):
        with mock.patch.dict("os.environ", {"CURSOR_HUB_HINT_PCT": "60"}):
            hint = _quick_action_for_context(60.0)
            assert hint is not None
            assert "Context is high" in hint.plain

    def test_hint_includes_source_label(self):
        hint = _quick_action_for_context(85.0, "#3")
        assert hint is not None
        assert "#3:" in hint.plain


class TestActionTodoRow:
    def test_combines_input_and_context_action(self):
        row = _action_todo_row(["#2"], _quick_action_for_context(85.0, "#2"))
        assert row is not None
        assert "Respond to #2" in row.plain
        assert "Context is high" in row.plain


class TestEtaHelpers:
    def test_eta_none_when_rate_not_growing(self):
        assert _eta_minutes_to_threshold(50.0, 0.0, 80.0) is None
        assert _eta_minutes_to_threshold(50.0, -2.0, 80.0) is None

    def test_eta_zero_when_already_over_threshold(self):
        assert _eta_minutes_to_threshold(82.0, 5.0, 80.0) == 0.0

    def test_eta_computation(self):
        # +10%/10m => +1% per minute; from 60% to 80% is 20 minutes.
        eta = _eta_minutes_to_threshold(60.0, 10.0, 80.0)
        assert eta == pytest.approx(20.0)

    def test_eta_format(self):
        assert _format_eta_minutes(0.5) == "<1m"
        assert _format_eta_minutes(17.2) == "17m"
        assert _format_eta_minutes(75.0) == "1h15m"
