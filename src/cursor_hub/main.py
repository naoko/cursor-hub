#!/usr/bin/env python3
"""
cursor-hub — Live terminal HUD for Cursor IDE
─────────────────────────────────────────────
Shows: active model, context bar, config counts, session timer

Usage:
  cursor-hub                           # live dashboard, refresh every 2s
  cursor-hub --once                    # print once and exit
  cursor-hub --once --json             # print machine-readable JSON snapshot
  cursor-hub --project /path/to/repo   # target a specific project dir
  cursor-hub --interval 5              # custom refresh rate (seconds)
  cursor-hub --show-paths              # show resolved settings path in PROJECT row
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live

# Allow `python src/cursor_hub/main.py` in addition to `python -m cursor_hub`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cursor_hub.config import count_mcps, count_notepads, count_rules_files
from cursor_hub.dashboard import cursor_process_info, make_panel, session_elapsed
from cursor_hub.paths import cursor_settings_path
from cursor_hub.state import (
    ComposerState,
    estimate_context_tokens,
    get_active_sessions,
    get_all_active_sessions,
    get_composer_state,
    get_model,
)

CONSOLE = Console()


def _positive_float(value: str) -> float:
    """argparse type: parse a strictly positive float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _session_to_json(session: ComposerState, index: int) -> dict:
    """Serialize a ComposerState for JSON output."""
    context_pct = None
    if session.context_limit > 0:
        context_pct = round((session.context_used / session.context_limit) * 100, 1)

    top_tools = sorted(session.tool_counts.items(), key=lambda x: -x[1])[:4]
    return {
        "index": index + 1,
        "name": session.name,
        "model": session.model,
        "mode": session.unified_mode,
        "force_mode": session.force_mode,
        "max_mode": session.max_mode,
        "context_used": session.context_used,
        "context_limit": session.context_limit,
        "context_pct": context_pct,
        "top_tools": [{"name": n, "count": c} for n, c in top_tools],
        "files_edited": session.files_edited,
        "subagents": session.subagents,
        "awaiting_user_input": session.awaiting_user_input,
        "pending_question_count": session.pending_question_count,
    }


def _build_json_snapshot(
    project_dir: Path,
    start_time: float,
    *,
    show_all: bool = False,
    show_paths: bool = False,
) -> dict:
    """Build a machine-readable snapshot of current cursor-hub state."""
    now = datetime.now().isoformat(timespec="seconds")
    elapsed_seconds = int(time.time() - start_time)
    is_running, mem_str = cursor_process_info()

    payload: dict[str, object] = {
        "timestamp": now,
        "elapsed_seconds": elapsed_seconds,
        "elapsed": session_elapsed(start_time),
        "project": str(project_dir),
        "config": {
            "rules": count_rules_files(project_dir),
            "mcps": count_mcps(project_dir),
            "notepads": count_notepads(project_dir),
        },
        "cursor_process": {
            "running": is_running,
            "memory": mem_str,
        },
    }

    if show_paths:
        sp = cursor_settings_path()
        if sp:
            payload["settings_path"] = str(sp)

    if show_all:
        all_sessions = get_all_active_sessions()
        projects: list[dict[str, object]] = []
        total_agents = 0
        for project_name, sessions in sorted(all_sessions.items()):
            agents = [_session_to_json(s, i) for i, s in enumerate(sessions)]
            total_agents += len(agents)
            projects.append(
                {
                    "name": project_name,
                    "agent_count": len(agents),
                    "agents": agents,
                }
            )
        payload["all_workspaces"] = {
            "project_count": len(projects),
            "agent_count": total_agents,
            "projects": projects,
        }
        return payload

    sessions = get_active_sessions(project_dir)
    if sessions:
        payload["agents"] = [_session_to_json(s, i) for i, s in enumerate(sessions)]
        payload["active_agent_count"] = len(sessions)
        return payload

    composer = get_composer_state(project_dir)
    model_str = get_model(project_dir)
    if composer.context_limit > 0:
        used, limit = composer.context_used, composer.context_limit
    else:
        used, limit = estimate_context_tokens(project_dir, model_str)
    context_pct = round((used / limit) * 100, 1) if limit else 0.0
    payload["model"] = model_str
    payload["mode"] = composer.unified_mode
    payload["context"] = {
        "used": used,
        "limit": limit,
        "pct": context_pct,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cursor-hub — live terminal HUD for Cursor IDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print dashboard once and exit",
    )
    parser.add_argument(
        "--project",
        default=".",
        metavar="DIR",
        help="Project directory to inspect (default: cwd)",
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=2.0,
        metavar="SECONDS",
        help="Refresh interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show sessions across all Cursor workspaces",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Show resolved settings path in PROJECT row",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON snapshot (use with --once)",
    )
    args = parser.parse_args()
    if args.json and not args.once:
        parser.error("--json currently requires --once")

    project_dir = Path(args.project).resolve()
    start_time = time.time()
    show_all = args.all
    show_paths = args.show_paths

    if args.once:
        if args.json:
            CONSOLE.print(
                json.dumps(
                    _build_json_snapshot(
                        project_dir,
                        start_time,
                        show_all=show_all,
                        show_paths=show_paths,
                    ),
                    indent=2,
                )
            )
            return
        CONSOLE.print(
            make_panel(
                project_dir,
                start_time,
                show_all=show_all,
                show_paths=show_paths,
            )
        )
        return

    try:
        with Live(
            make_panel(
                project_dir,
                start_time,
                show_all=show_all,
                show_paths=show_paths,
            ),
            console=CONSOLE,
            refresh_per_second=1.0 / args.interval,
            screen=False,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(
                    make_panel(
                        project_dir,
                        start_time,
                        show_all=show_all,
                        show_paths=show_paths,
                    )
                )
    except KeyboardInterrupt:
        CONSOLE.print("\n[dim]cursor-hub stopped[/dim]")


if __name__ == "__main__":
    main()
