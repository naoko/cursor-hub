"""Additional coverage-focused tests for cursor-hub."""

import base64
import json
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from cursor_hub import config, dashboard, main, paths, state
from cursor_hub.state import ComposerState
from cursor_hub.models import classify_model, context_window_for


def _make_db(path: Path, stmts: list[tuple[str, tuple]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)"
    )
    for sql, args in stmts:
        conn.execute(sql, args)
    conn.commit()
    conn.close()


class TestPathsAndConfig:
    def test_os_specific_paths(self, tmp_path):
        home = tmp_path
        # Darwin
        darwin_settings = (
            home
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "settings.json"
        )
        darwin_settings.parent.mkdir(parents=True)
        darwin_settings.write_text("{}")
        with mock.patch("cursor_hub.paths.platform.system", return_value="Darwin"):
            with mock.patch("cursor_hub.paths.Path.home", return_value=home):
                assert paths.cursor_settings_path() == darwin_settings
                assert "globalStorage/state.vscdb" in str(paths.state_db_path())
        # Linux
        linux_settings = home / ".config" / "Cursor" / "User" / "settings.json"
        linux_settings.parent.mkdir(parents=True, exist_ok=True)
        linux_settings.write_text("{}")
        with mock.patch("cursor_hub.paths.platform.system", return_value="Linux"):
            with mock.patch("cursor_hub.paths.Path.home", return_value=home):
                assert paths.cursor_settings_path() == linux_settings
        # Windows
        appdata = home / "AppData" / "Roaming"
        win_settings = appdata / "Cursor" / "User" / "settings.json"
        win_settings.parent.mkdir(parents=True, exist_ok=True)
        win_settings.write_text("{}")
        with mock.patch("cursor_hub.paths.platform.system", return_value="Windows"):
            with mock.patch.dict("os.environ", {"APPDATA": str(appdata)}):
                assert paths.cursor_settings_path() == win_settings
                ws = appdata / "Cursor" / "User" / "workspaceStorage"
                ws.mkdir(parents=True, exist_ok=True)
                assert paths.cursor_workspace_storage() == ws

    def test_jsonc_and_config_counts(self, tmp_path):
        jsonc = tmp_path / "x.json"
        jsonc.write_text('{"a": 1, // c\n "b": "v/*ok*/"}')
        assert paths.read_json(jsonc)["a"] == 1
        # rules
        (tmp_path / ".cursorrules").write_text("legacy")
        prules = tmp_path / ".cursor" / "rules"
        prules.mkdir(parents=True)
        (prules / "a.mdc").write_text("x")
        home = tmp_path / "home"
        grules = home / ".cursor" / "rules"
        grules.mkdir(parents=True)
        (grules / "g.mdc").write_text("x")
        with mock.patch("cursor_hub.config.Path.home", return_value=home):
            assert config.count_rules_files(tmp_path) == 3
        # mcps + notepads
        (tmp_path / ".cursor" / "mcp.json").write_text(
            json.dumps({"mcpServers": {"a": {}}})
        )
        (home / ".cursor").mkdir(parents=True, exist_ok=True)
        (home / ".cursor" / "mcp.json").write_text(
            json.dumps({"mcpServers": {"b": {}}})
        )
        np = tmp_path / ".cursor" / "notepads"
        np.mkdir(parents=True, exist_ok=True)
        (np / "n.md").write_text("note")
        with mock.patch("cursor_hub.config.Path.home", return_value=home):
            assert config.count_mcps(tmp_path) == 2
        assert config.count_notepads(tmp_path) == 1

    def test_config_list_mcps_and_empty_notepads(self, tmp_path):
        (tmp_path / ".cursor").mkdir(parents=True)
        (tmp_path / ".cursor" / "mcp.json").write_text(
            json.dumps({"mcpServers": [{}, {}]})
        )
        assert config.count_mcps(tmp_path) == 2
        assert config.count_notepads(tmp_path) == 0

    def test_read_json_failure(self, tmp_path):
        assert paths.read_json(tmp_path / "missing.json") == {}

    def test_paths_remaining_branches(self, tmp_path):
        home = tmp_path
        with mock.patch("cursor_hub.paths.platform.system", return_value="Linux"):
            with mock.patch("cursor_hub.paths.Path.home", return_value=home):
                ws = home / ".config" / "Cursor" / "User" / "workspaceStorage"
                ws.mkdir(parents=True)
                assert paths.cursor_workspace_storage() == ws
                assert ".config/Cursor/User/globalStorage/state.vscdb" in str(
                    paths.state_db_path()
                )

        appdata = home / "AppData" / "Roaming"
        with mock.patch("cursor_hub.paths.platform.system", return_value="Windows"):
            with mock.patch.dict("os.environ", {"APPDATA": str(appdata)}):
                assert "AppData/Roaming/Cursor/User/globalStorage/state.vscdb" in str(
                    paths.state_db_path()
                )

        bad = tmp_path / "bad.json"
        bad.write_text("{ invalid /*x*/ // y")
        assert paths.read_json(bad) == {}


class TestModelsBranches:
    def test_classify_additional_families(self):
        assert "o3" in classify_model("o3-mini")[0]
        assert "Gemini" in classify_model("gemini-2.0-flash")[0]
        assert "Cursor Small" in classify_model("cursor-small")[0]
        assert "Default" in classify_model("default")[0]
        label, color = classify_model("mystery-model-x")
        assert "mystery-model-x" in label
        assert color == "bright_white"

    def test_context_window_family_fallbacks(self):
        assert context_window_for("gpt-4.1") == 1_047_576
        assert context_window_for("gpt-4o") == 128_000
        assert context_window_for("o4-mini") == 200_000
        assert context_window_for("gemini-1.5-pro") == 2_000_000
        assert context_window_for("gemini-2.0-flash") == 1_048_576
        assert context_window_for("grok-3") == 131_072
        assert context_window_for("deepseek-r1") == 128_000
        assert context_window_for("unknown-model") == 128_000


class TestDashboardBuild:
    def test_dashboard_helpers_cover_edges(self):
        assert dashboard._truncate("abc", 2) == "a…"
        assert dashboard._compact_tool_name("ReadFile") == "RF"
        assert dashboard._compact_tool_name("Unknown") == "Unknown"
        assert dashboard._format_tokens_compact(999) == "999"
        assert dashboard._format_tokens_compact(1_500_000) == "1.5M"
        assert dashboard._quick_action_for_context(None) is None
        assert dashboard._quick_action_for_context(95.0) is not None
        no_data = dashboard._context_bar(0, 1000).plain
        assert "no active session data" in no_data

        # process info: running + fallback
        proc = mock.Mock()
        proc.info = {"name": "Cursor", "memory_info": mock.Mock(rss=100 * 1_048_576)}
        with mock.patch(
            "cursor_hub.dashboard.psutil.process_iter", return_value=[proc]
        ):
            running, mem = dashboard.cursor_process_info()
            assert running and mem == "100 MB"
        with mock.patch(
            "cursor_hub.dashboard.psutil.process_iter", side_effect=Exception("x")
        ):
            running, _ = dashboard.cursor_process_info()
            assert running is False

    def test_build_dashboard_and_panel(self, tmp_path):
        s = ComposerState(
            name="session",
            model="claude-sonnet-4",
            unified_mode="agent",
            force_mode="edit",
            context_used=80_000,
            context_limit=100_000,
            tool_counts={"ReadFile": 3, "Shell": 2},
            files_edited=["a.py"],
            subagents=[{"type": "explore", "description": "scan"}],
        )
        with mock.patch("cursor_hub.dashboard.get_active_sessions", return_value=[s]):
            with mock.patch("cursor_hub.dashboard.count_rules_files", return_value=1):
                with mock.patch("cursor_hub.dashboard.count_mcps", return_value=2):
                    with mock.patch(
                        "cursor_hub.dashboard.count_notepads", return_value=3
                    ):
                        with mock.patch(
                            "cursor_hub.dashboard.cursor_process_info",
                            return_value=(True, "123 MB"),
                        ):
                            table = dashboard.build_dashboard(
                                tmp_path, 0.0, show_paths=False
                            )
                            assert table is not None
                            panel = dashboard.make_panel(tmp_path, 0.0)
                            assert panel is not None
        s.awaiting_user_input = True
        row = dashboard._format_session_row(s, 0, history_key="k")
        assert "awaiting input" in row.plain

    def test_build_all_dashboard_empty(self):
        with mock.patch(
            "cursor_hub.dashboard.get_all_active_sessions", return_value={}
        ):
            with mock.patch(
                "cursor_hub.dashboard.cursor_process_info", return_value=(False, "—")
            ):
                table = dashboard.build_all_dashboard(0.0)
                assert table is not None

    def test_build_dashboard_no_sessions_fallback(self, tmp_path):
        composer = ComposerState(
            unified_mode="chat", context_used=100, context_limit=1000
        )
        with mock.patch("cursor_hub.dashboard.get_active_sessions", return_value=[]):
            with mock.patch(
                "cursor_hub.dashboard.get_composer_state", return_value=composer
            ):
                with mock.patch(
                    "cursor_hub.dashboard.get_model", return_value="gpt-4o"
                ):
                    with mock.patch(
                        "cursor_hub.dashboard.count_rules_files", return_value=0
                    ):
                        with mock.patch(
                            "cursor_hub.dashboard.count_mcps", return_value=0
                        ):
                            with mock.patch(
                                "cursor_hub.dashboard.count_notepads", return_value=0
                            ):
                                with mock.patch(
                                    "cursor_hub.dashboard.cursor_process_info",
                                    return_value=(False, "—"),
                                ):
                                    t = dashboard.build_dashboard(
                                        tmp_path, 0.0, show_paths=True
                                    )
                                    assert t is not None

    def test_build_all_dashboard_with_projects(self):
        s = ComposerState(
            model="gpt-4o",
            unified_mode="agent",
            context_used=85,
            context_limit=100,
            composer_id="x",
        )
        with mock.patch(
            "cursor_hub.dashboard.get_all_active_sessions", return_value={"p1": [s]}
        ):
            with mock.patch(
                "cursor_hub.dashboard.cursor_process_info", return_value=(True, "1 MB")
            ):
                t = dashboard.build_all_dashboard(0.0)
                assert t is not None


class TestStateDeepPaths:
    def test_extract_tool_usage_and_estimate_tokens(self, tmp_path):
        db = tmp_path / "state.vscdb"
        h = "ab" * 32
        raw = bytes([0x0A, 0x20]) + bytes.fromhex(h)
        cs = "~" + base64.b64encode(raw).decode()
        blob = {
            "content": [
                {"type": "tool-call", "toolName": "ReadFile", "args": {}},
                {
                    "type": "tool-call",
                    "toolName": "Write",
                    "args": {"path": "/tmp/f.py"},
                },
                {
                    "type": "tool-call",
                    "toolName": "Task",
                    "args": {"subagent_type": "explore", "description": "scan"},
                },
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                )
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        tc, fe, sa, pending = state._extract_tool_usage({"conversationState": cs}, cur)
        conn.close()
        assert tc["ReadFile"] == 1
        assert fe == ["f.py"]
        assert sa[0]["type"] == "explore"
        assert pending == 0

        # estimate_context_tokens path
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "workspace.json").write_text("{}")
        (ws / "composer.json").write_text("x" * 400)
        with mock.patch("cursor_hub.state._find_workspace_folder", return_value=ws):
            used, limit = state.estimate_context_tokens(tmp_path, "claude-sonnet-4")
            assert used > 0
            assert limit == 200_000

    def test_extract_tool_usage_pending_ask_question(self, tmp_path):
        db = tmp_path / "pending.vscdb"
        h = "cd" * 32
        raw = bytes([0x0A, 0x20]) + bytes.fromhex(h)
        cs = "~" + base64.b64encode(raw).decode()
        blob = {
            "content": [
                {
                    "type": "tool-call",
                    "toolName": "AskQuestion",
                    "toolCallId": "abc123",
                    "args": {"questions": []},
                }
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                )
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        _, _, _, pending = state._extract_tool_usage({"conversationState": cs}, cur)
        conn.close()
        assert pending == 1

    def test_pending_questions_from_bubbles(self, tmp_path):
        db = tmp_path / "bubbles.vscdb"
        cid = "cid-xyz"
        unanswered = {
            "createdAt": "2026-03-29T00:00:10.000Z",
            "type": 2,
            "toolFormerData": {"name": "ask_question", "status": "completed"},
        }
        assistant_followup = {
            "createdAt": "2026-03-29T00:00:20.000Z",
            "type": 2,
            "text": "Got it, thanks.",
        }
        answered_q = {
            "createdAt": "2026-03-29T00:00:15.000Z",
            "type": 2,
            "toolFormerData": {"name": "ask_question", "status": "completed"},
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"bubbleId:{cid}:q1", json.dumps(unanswered)),
                ),
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"bubbleId:{cid}:q2", json.dumps(answered_q)),
                ),
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"bubbleId:{cid}:a", json.dumps(assistant_followup)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        # Assistant follow-up at :20 marks earlier structured asks as answered.
        assert state._pending_questions_from_bubbles(cur, cid) == 0
        conn.close()

        # New unanswered question after assistant follow-up.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                f"bubbleId:{cid}:q3",
                json.dumps(
                    {
                        "createdAt": "2026-03-29T00:00:30.000Z",
                        "type": 2,
                        "toolFormerData": {
                            "name": "ask_question",
                            "status": "completed",
                        },
                    }
                ),
            ),
        )
        conn.commit()
        cur = conn.cursor()
        assert state._pending_questions_from_bubbles(cur, cid) == 1
        conn.close()

    def test_get_all_active_sessions(self, tmp_path):
        home = tmp_path
        ws_root = (
            home
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "workspaceStorage"
        )
        ws = ws_root / "w1"
        ws.mkdir(parents=True)
        (ws / "workspace.json").write_text(json.dumps({"folder": "file:///tmp/myproj"}))
        ws_db = ws / "state.vscdb"
        item = {
            "selectedComposerIds": ["cid-1"],
            "allComposers": [{"composerId": "cid-1", "name": "A"}],
        }
        _make_db(
            ws_db,
            [
                (
                    "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                    ("composer.composerData", json.dumps(item)),
                )
            ],
        )
        gdb = (
            home
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "globalStorage"
            / "state.vscdb"
        )
        gdb.parent.mkdir(parents=True)
        data = {
            "name": "A",
            "modelConfig": {"modelName": "gpt-4o"},
            "unifiedMode": "agent",
            "contextTokensUsed": 1,
            "contextTokenLimit": 2,
        }
        _make_db(
            gdb,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    ("composerData:cid-1", json.dumps(data)),
                )
            ],
        )
        with mock.patch("cursor_hub.paths.platform.system", return_value="Darwin"):
            with mock.patch("cursor_hub.paths.Path.home", return_value=home):
                result = state.get_all_active_sessions()
        assert "myproj" in result
        assert result["myproj"][0].model == "gpt-4o"

    def test_state_edge_paths(self, tmp_path):
        # no workspace storage
        with mock.patch("cursor_hub.state.cursor_workspace_storage", return_value=None):
            assert state.get_all_active_sessions() == {}

        # get_composer_state fallback when db missing
        with mock.patch(
            "cursor_hub.state.state_db_path", return_value=tmp_path / "none.db"
        ):
            s = state.get_composer_state()
            assert isinstance(s, ComposerState)

        # find workspace folder URI decode
        ws = tmp_path / "ws"
        ws.mkdir()
        folder = ws / "a"
        folder.mkdir()
        (folder / "workspace.json").write_text(
            json.dumps({"folder": f"file://{tmp_path}"})
        )
        with mock.patch("cursor_hub.state.cursor_workspace_storage", return_value=ws):
            found = state._find_workspace_folder(tmp_path)
            assert found == folder


class TestMainExecutionPaths:
    def test_positive_float_error(self):
        with pytest.raises(Exception):
            main._positive_float("x")

    def test_main_once_json_and_error(self, tmp_path, capsys):
        with mock.patch(
            "cursor_hub.main._build_json_snapshot", return_value={"ok": True}
        ):
            with mock.patch("sys.argv", ["cursor-hub", "--once", "--json"]):
                main.main()
        out = capsys.readouterr().out
        assert '"ok": true' in out

        with mock.patch("sys.argv", ["cursor-hub", "--json"]):
            with pytest.raises(SystemExit):
                main.main()

    def test_main_live_keyboard_interrupt(self, tmp_path):
        class _LiveCtx:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, _):
                return None

        with mock.patch("cursor_hub.main.Live", _LiveCtx):
            with mock.patch("cursor_hub.main.make_panel", return_value="PANEL"):
                with mock.patch(
                    "cursor_hub.main.time.sleep", side_effect=[None, KeyboardInterrupt]
                ):
                    with mock.patch("sys.argv", ["cursor-hub"]):
                        main.main()

    def test_build_json_snapshot_show_all_and_paths(self, tmp_path):
        s = ComposerState(
            model="gpt-4o", unified_mode="agent", context_used=10, context_limit=100
        )
        with mock.patch("cursor_hub.main.count_rules_files", return_value=1):
            with mock.patch("cursor_hub.main.count_mcps", return_value=2):
                with mock.patch("cursor_hub.main.count_notepads", return_value=3):
                    with mock.patch(
                        "cursor_hub.main.cursor_process_info",
                        return_value=(True, "10 MB"),
                    ):
                        with mock.patch(
                            "cursor_hub.main.cursor_settings_path",
                            return_value=Path("/tmp/s.json"),
                        ):
                            with mock.patch(
                                "cursor_hub.main.get_all_active_sessions",
                                return_value={"p": [s]},
                            ):
                                payload = main._build_json_snapshot(
                                    tmp_path, 0.0, show_all=True, show_paths=True
                                )
        assert payload["settings_path"] == "/tmp/s.json"
        assert payload["all_workspaces"]["project_count"] == 1

    def test_main_once_panel_path(self):
        with mock.patch("cursor_hub.main.make_panel", return_value="PANEL"):
            with mock.patch("sys.argv", ["cursor-hub", "--once"]):
                main.main()


def test___main___module_executes():
    with mock.patch("cursor_hub.main.main") as m:
        import importlib
        import cursor_hub.__main__ as mod

        importlib.reload(mod)
        assert m.called


class TestExtractToolUsageEdgeCases:
    """Cover edge cases in _extract_tool_usage."""

    def _make_conversation_state(self, h_hex: str) -> str:
        """Build a base64-encoded conversationState containing one hash."""
        raw = bytes([0x0A, 0x20]) + bytes.fromhex(h_hex)
        return "~" + base64.b64encode(raw).decode()

    def test_empty_conversation_state(self, tmp_path):
        db = tmp_path / "empty_cs.vscdb"
        _make_db(db, [])
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        tc, fe, sa, p = state._extract_tool_usage({}, cur)
        assert tc == {} and fe == [] and sa == [] and p == 0
        # Also test empty string
        tc2, _, _, _ = state._extract_tool_usage({"conversationState": ""}, cur)
        assert tc2 == {}
        conn.close()

    def test_invalid_base64(self, tmp_path):
        db = tmp_path / "bad64.vscdb"
        _make_db(db, [])
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        tc, fe, sa, p = state._extract_tool_usage(
            {"conversationState": "!!!not-base64!!!"}, cur
        )
        assert tc == {} and fe == [] and sa == [] and p == 0
        conn.close()

    def test_no_hashes_in_protobuf(self, tmp_path):
        db = tmp_path / "nohash.vscdb"
        _make_db(db, [])
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        # Valid base64 but no 0x0A 0x20 pattern
        cs = base64.b64encode(b"short data no hashes").decode()
        tc, fe, sa, p = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc == {}
        conn.close()

    def test_blob_not_found_in_db(self, tmp_path):
        db = tmp_path / "noblob.vscdb"
        h = "ee" * 32
        _make_db(db, [])
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, fe, sa, p = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc == {}
        conn.close()

    def test_non_dict_content_items_skipped(self, tmp_path):
        db = tmp_path / "nondict.vscdb"
        h = "aa" * 32
        blob = {"content": ["not a dict", 42, None]}
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, _, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc == {}
        conn.close()

    def test_tool_call_with_no_name_skipped(self, tmp_path):
        db = tmp_path / "noname.vscdb"
        h = "bb" * 32
        blob = {"content": [{"type": "tool-call"}]}
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, _, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc == {}
        conn.close()

    def test_non_dict_args_handled(self, tmp_path):
        db = tmp_path / "badargs.vscdb"
        h = "cc" * 32
        blob = {
            "content": [
                {"type": "tool-call", "toolName": "Write", "args": "not a dict"},
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, fe, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc["Write"] == 1
        assert fe == []  # no file extracted from non-dict args
        conn.close()

    def test_tool_result_ask_question_resolves_pending(self, tmp_path):
        db = tmp_path / "askresolved.vscdb"
        h = "dd" * 32
        blob = {
            "content": [
                {
                    "type": "tool-call",
                    "toolName": "AskQuestion",
                    "toolCallId": "q1",
                    "args": {},
                },
                {
                    "type": "tool-result",
                    "toolName": "AskQuestion",
                    "toolCallId": "q1",
                },
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        _, _, _, pending = state._extract_tool_usage({"conversationState": cs}, cur)
        assert pending == 0
        conn.close()

    def test_corrupt_blob_json_skipped(self, tmp_path):
        db = tmp_path / "corruptblob.vscdb"
        h = "ff" * 32
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", "not valid json{{{"),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, _, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc == {}
        conn.close()

    def test_blob_stored_as_bytes(self, tmp_path):
        """Cover the bytes decode branch (blob_row[0] not isinstance str)."""
        db = tmp_path / "bytesblob.vscdb"
        h = "11" * 32
        blob_bytes = json.dumps(
            {"content": [{"type": "tool-call", "toolName": "Shell", "args": {}}]}
        ).encode()
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)"
        )
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (f"agentKv:blob:{h}", blob_bytes),
        )
        conn.commit()
        conn.close()
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, _, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert tc.get("Shell") == 1
        conn.close()

    def test_str_replace_tracked_as_file_edit(self, tmp_path):
        db = tmp_path / "strreplace.vscdb"
        h = "22" * 32
        blob = {
            "content": [
                {
                    "type": "tool-call",
                    "toolName": "StrReplace",
                    "args": {"filePath": "/tmp/foo/bar.ts"},
                },
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        tc, fe, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert "bar.ts" in fe
        conn.close()

    def test_duplicate_file_edits_deduped(self, tmp_path):
        db = tmp_path / "dedup.vscdb"
        h = "33" * 32
        blob = {
            "content": [
                {
                    "type": "tool-call",
                    "toolName": "Write",
                    "args": {"path": "/a/b.py"},
                },
                {
                    "type": "tool-call",
                    "toolName": "Write",
                    "args": {"path": "/c/b.py"},
                },
            ]
        }
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (f"agentKv:blob:{h}", json.dumps(blob)),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cs = self._make_conversation_state(h)
        _, fe, _, _ = state._extract_tool_usage({"conversationState": cs}, cur)
        assert fe == ["b.py"]
        conn.close()


class TestPendingQuestionsBubblesEdge:
    def test_bubbles_db_error_returns_zero(self, tmp_path):
        db = tmp_path / "bad.vscdb"
        _make_db(db, [])
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        # Drop the table to trigger an exception in the LIKE query
        conn.execute("DROP TABLE cursorDiskKV")
        conn.commit()
        result = state._pending_questions_from_bubbles(cur, "cid")
        assert result == 0
        conn.close()

    def test_bubbles_no_ask_times(self, tmp_path):
        db = tmp_path / "noask.vscdb"
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (
                        "bubbleId:cid:b1",
                        json.dumps(
                            {"createdAt": "2026-01-01", "type": 2, "text": "hi"}
                        ),
                    ),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        assert state._pending_questions_from_bubbles(cur, "cid") == 0
        conn.close()

    def test_bubbles_invalid_json(self, tmp_path):
        db = tmp_path / "badjson.vscdb"
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    ("bubbleId:cid:b1", "not json"),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        assert state._pending_questions_from_bubbles(cur, "cid") == 0
        conn.close()

    def test_bubbles_no_created_at(self, tmp_path):
        db = tmp_path / "nocreated.vscdb"
        _make_db(
            db,
            [
                (
                    "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
                    (
                        "bubbleId:cid:b1",
                        json.dumps(
                            {"type": 2, "toolFormerData": {"name": "ask_question"}}
                        ),
                    ),
                ),
            ],
        )
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        assert state._pending_questions_from_bubbles(cur, "cid") == 0
        conn.close()


class TestDashboardTruncateEdge:
    def test_truncate_max_len_one(self):
        assert dashboard._truncate("abc", 1) == "a"

    def test_truncate_max_len_zero(self):
        assert dashboard._truncate("abc", 0) == ""


class TestEstimateContextTokensEdges:
    def test_no_workspace_returns_zero(self, tmp_path):
        with mock.patch("cursor_hub.state._find_workspace_folder", return_value=None):
            used, limit = state.estimate_context_tokens(tmp_path, "gpt-4o")
            assert used == 0
            assert limit == 128_000

    def test_skips_workspace_and_settings_json(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "workspace.json").write_text("{}")
        (ws / "settings.json").write_text("{}")
        (ws / "other.json").write_text("x" * 800)
        with mock.patch("cursor_hub.state._find_workspace_folder", return_value=ws):
            used, _ = state.estimate_context_tokens(tmp_path, "gpt-4o")
            assert used == 200  # 800 / 4

    def test_counted_paths_not_double_counted(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "workspace.json").write_text("{}")
        (ws / "composer.json").write_text("x" * 400)
        with mock.patch("cursor_hub.state._find_workspace_folder", return_value=ws):
            used, _ = state.estimate_context_tokens(tmp_path, "gpt-4o")
            assert used == 100  # 400 / 4, not double-counted


class TestFindWorkspaceEdges:
    def test_normalize_non_file_uri(self, tmp_path):
        ws = tmp_path / "ws"
        folder = ws / "a"
        folder.mkdir(parents=True)
        (folder / "workspace.json").write_text(json.dumps({"folder": str(tmp_path)}))
        with mock.patch("cursor_hub.state.cursor_workspace_storage", return_value=ws):
            found = state._find_workspace_folder(tmp_path)
            assert found == folder

    def test_empty_folder_value(self, tmp_path):
        ws = tmp_path / "ws"
        folder = ws / "a"
        folder.mkdir(parents=True)
        (folder / "workspace.json").write_text(json.dumps({"folder": ""}))
        with mock.patch("cursor_hub.state.cursor_workspace_storage", return_value=ws):
            found = state._find_workspace_folder(tmp_path)
            assert found is None


class TestGetModelFromSettingsNested:
    def test_nested_empty_fallback(self):
        settings = {"cursor": {"chat": {}}}
        assert state.get_model_from_settings(settings) == ""


class TestModelNormalizeSuffix:
    def test_strips_thinking_and_preview(self):
        from cursor_hub.models import _normalize_model_suffix

        assert _normalize_model_suffix("claude-opus-4-thinking") == "claude-opus-4"
        assert _normalize_model_suffix("gpt-4o-preview") == "gpt-4o"
        assert _normalize_model_suffix("model-default") == "model"
        assert _normalize_model_suffix("model-latest") == "model"

    def test_classify_haiku(self):
        from cursor_hub.models import classify_model

        label, color = classify_model("claude-haiku-3.5")
        assert "Haiku" in label
        assert color == "bright_blue"

    def test_classify_deepseek(self):
        label, color = classify_model("deepseek-v3")
        assert "DeepSeek" in label or "deepseek" in label

    def test_classify_grok(self):
        label, color = classify_model("grok-3")
        assert "Grok" in label or "grok" in label

    def test_context_gemini_generic(self):
        assert context_window_for("gemini-2.5-pro") == 1_000_000
