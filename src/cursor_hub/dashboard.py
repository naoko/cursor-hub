"""Rich terminal rendering for the cursor-hub dashboard."""

import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import count_mcps, count_notepads, count_rules_files
from .models import classify_model
from .paths import cursor_settings_path
from .state import (
    ComposerState,
    estimate_context_tokens,
    get_active_sessions,
    get_all_active_sessions,
    get_composer_state,
    get_model,
)

BLOCK_FULL = "█"
BLOCK_EMPTY = "░"
BAR_WIDTH = 12
TREND_WINDOW_SECONDS = 600.0
_CONTEXT_HISTORY: dict[str, deque[tuple[float, float]]] = {}

_MODE_LABELS: dict[str, tuple[str, str]] = {
    "agent": ("Agent", "bright_green"),
    "chat": ("Chat", "bright_cyan"),
    "edit": ("Edit", "bright_yellow"),
    "plan": ("Plan", "bright_magenta"),
}

_TOOL_ALIASES: dict[str, str] = {
    "ReadFile": "RF",
    "ApplyPatch": "AP",
    "TodoWrite": "TW",
    "Subagent": "SA",
    "Shell": "SH",
    "ReadLints": "RL",
}


def session_elapsed(start_time: float) -> str:
    """Format elapsed time since start_time as a human-readable string."""
    elapsed = int(time.time() - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def cursor_process_info() -> tuple[bool, str]:
    """Return (is_running, total RSS memory) for Cursor processes."""
    try:
        total_mb = 0.0
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = (proc.info.get("name") or "").lower()
            if "cursor" in name:
                mem = proc.info["memory_info"]
                if mem:
                    total_mb += mem.rss / 1_048_576
        if total_mb > 0:
            return True, f"{total_mb:.0f} MB"
    except Exception:
        pass
    return False, "—"


def _context_bar(used: int, max_tokens: int) -> Text:
    pct = (used / max_tokens) if max_tokens else 0
    filled = round(pct * BAR_WIDTH)
    filled = max(0, min(BAR_WIDTH, filled))

    color = "bright_green" if pct < 0.6 else ("yellow" if pct < 0.8 else "bright_red")

    bar = Text()
    bar.append(BLOCK_FULL * filled, style=f"bold {color}")
    bar.append(BLOCK_EMPTY * (BAR_WIDTH - filled), style="dim white")
    bar.append(f"  {pct * 100:4.1f}%", style=color)

    if used:
        bar.append(
            f"  {_format_tokens_compact(used)} / {_format_tokens_compact(max_tokens)} tokens",
            style="dim white",
        )
    else:
        bar.append("  (no active session data)", style="dim white")
    return bar


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis when needed."""
    if len(text) <= max_len:
        return text
    if max_len <= 1:
        return text[:max_len]
    return text[: max_len - 1] + "…"


def _compact_tool_name(name: str) -> str:
    """Shorten common tool names to preserve horizontal space."""
    return _TOOL_ALIASES.get(name, name)


def _format_tokens_compact(value: int) -> str:
    """Format token counts in compact K/M notation."""
    abs_value = abs(value)
    if abs_value < 1_000:
        return str(value)
    if abs_value < 1_000_000:
        compact = value / 1_000
        suffix = "K"
    else:
        compact = value / 1_000_000
        suffix = "M"

    if abs(compact) >= 100 or float(compact).is_integer():
        return f"{compact:.0f}{suffix}"
    return f"{compact:.1f}{suffix}"


def _record_context_pct(
    history_key: str, pct: float, now: float
) -> deque[tuple[float, float]]:
    """Append a context percentage sample and prune old history."""
    history = _CONTEXT_HISTORY.setdefault(history_key, deque())
    history.append((now, pct))
    cutoff = now - TREND_WINDOW_SECONDS
    while history and history[0][0] < cutoff:
        history.popleft()
    return history


def _trend_delta_per_10m(
    history: deque[tuple[float, float]], current_pct: float, now: float
) -> float | None:
    """Return projected context delta per 10 minutes from rolling history."""
    if len(history) < 2:
        return None
    oldest_ts, oldest_pct = history[0]
    span = now - oldest_ts
    # Avoid noisy spikes with too-little history.
    if span < 30:
        return None
    return (current_pct - oldest_pct) * (TREND_WINDOW_SECONDS / span)


def _eta_minutes_to_threshold(
    current_pct: float, delta_per_10m: float, threshold_pct: float
) -> float | None:
    """Estimate minutes to reach threshold from current trend slope."""
    if current_pct >= threshold_pct:
        return 0.0
    # delta_per_10m is in percentage points per 10 minutes.
    rate_per_min = delta_per_10m / 10.0
    if rate_per_min <= 0:
        return None
    return (threshold_pct - current_pct) / rate_per_min


def _format_eta_minutes(minutes: float) -> str:
    """Format ETA minutes as compact human text."""
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{int(round(minutes))}m"
    hours = int(minutes // 60)
    mins = int(round(minutes - (hours * 60)))
    if mins == 60:
        hours += 1
        mins = 0
    return f"{hours}h{mins:02d}m"


def _append_trend_hint(text: Text, history_key: str, pct: float) -> None:
    """Append a signed trend hint like +4.0%/10m when enough history exists."""
    now = time.time()
    history = _record_context_pct(history_key, pct, now)
    delta = _trend_delta_per_10m(history, pct, now)
    if delta is None:
        return
    # Hide very small projected movement to avoid noisy 0.0%/10m hints.
    if abs(delta) < 0.05:
        return
    sign = "+" if delta >= 0 else ""
    trend_color = "dim white"
    decimals = 2 if abs(delta) < 1 else 1
    text.append(f"  {sign}{delta:.{decimals}f}%/10m", style=trend_color)
    eta80 = _eta_minutes_to_threshold(pct, delta, 80.0)
    eta90 = _eta_minutes_to_threshold(pct, delta, 90.0)
    if eta80 is not None and pct < 80:
        text.append(f" ETA80:{_format_eta_minutes(eta80)}", style="dim white")
    if eta90 is not None and pct < 90:
        text.append(f" ETA90:{_format_eta_minutes(eta90)}", style="dim white")


def _quick_action_for_context(
    max_pct: float | None, source_label: str | None = None
) -> Text | None:
    """Return a contextual action hint when context usage is high."""
    if max_pct is None:
        return None
    warn_threshold = 80.0
    override = os.environ.get("CURSOR_HUB_HINT_PCT", "").strip()
    if override:
        try:
            parsed = float(override)
            if 0 <= parsed <= 100:
                warn_threshold = parsed
        except ValueError:
            pass

    urgent_threshold = max(90.0, warn_threshold + 10.0)
    prefix = f"{source_label}: " if source_label else ""
    if max_pct >= urgent_threshold:
        return Text(
            f"{prefix}High context usage. Start a new session now and paste a short handoff summary.",
            style="bold bright_red",
        )
    if max_pct >= warn_threshold:
        return Text(
            f"{prefix}Context is high. Consider starting a new session and pasting a handoff summary.",
            style="yellow",
        )
    return None


def _action_todo_row(
    awaiting_sources: list[str], context_action: Text | None
) -> Text | None:
    """Build a TODO-style ACTION row from pending input + context recommendation."""
    items: list[str] = []
    if awaiting_sources:
        shown = ", ".join(awaiting_sources[:3])
        if len(awaiting_sources) > 3:
            shown += f" +{len(awaiting_sources) - 3}"
        items.append(f"Respond to {shown}")
    if context_action:
        items.append(context_action.plain)
    if not items:
        return None

    row = Text()
    for i, item in enumerate(items):
        if i > 0:
            row.append("\n", style="dim white")
        style = "bold yellow" if i == 0 and awaiting_sources else "yellow"
        row.append("• ", style=style)
        row.append(item, style=style)
    return row


def _pill(icon: str, count: int, label: str, color: str) -> Text:
    t = Text()
    t.append(f" {icon} {count} {label} ", style=f"{color} on grey11")
    return t


def _format_mode(composer: ComposerState) -> Text:
    """Format the Composer mode line from state."""
    t = Text()

    if not composer.unified_mode:
        t.append("—", style="dim white")
        return t

    mode_label, mode_color = _MODE_LABELS.get(
        composer.unified_mode, (composer.unified_mode.title(), "bright_white")
    )
    t.append(mode_label, style=f"bold {mode_color}")

    if composer.force_mode and composer.force_mode != composer.unified_mode:
        fm_label, fm_color = _MODE_LABELS.get(
            composer.force_mode, (composer.force_mode.title(), "bright_white")
        )
        t.append(" + ", style="dim white")
        t.append(fm_label, style=fm_color)

    if composer.max_mode:
        t.append("  MAX", style="bold bright_red")

    return t


def _format_session_row(
    session: ComposerState, index: int, *, history_key: str
) -> Text:
    """Format a single active session line."""
    t = Text()

    t.append(f"#{index + 1} ", style="bold bright_white")

    label, color = classify_model(session.model)
    t.append(label, style=f"bold {color}")

    t.append("  ", style="dim white")
    mode_text = _format_mode(session)
    t.append_text(mode_text)

    if session.context_limit > 0:
        pct = (session.context_used / session.context_limit) * 100
        ctx_color = (
            "bright_green" if pct < 60 else ("yellow" if pct < 80 else "bright_red")
        )
        t.append(f"  {pct:.0f}%", style=ctx_color)
        t.append(
            f" ({_format_tokens_compact(session.context_used)}/"
            f"{_format_tokens_compact(session.context_limit)})",
            style="dim white",
        )
        _append_trend_hint(t, history_key, pct)

    if session.name:
        session_name = _truncate(session.name, 32)
        t.append(f'  "{session_name}"', style="dim bright_black")

    if session.tool_counts:
        top_tools = sorted(session.tool_counts.items(), key=lambda x: -x[1])[:3]
        tool_str = " ".join(f"{_compact_tool_name(n)}:{c}" for n, c in top_tools)
        t.append(f"  [{tool_str}]", style="dim cyan")

    if session.files_edited:
        files_str = ", ".join(session.files_edited[:2])
        if len(session.files_edited) > 2:
            files_str += f" +{len(session.files_edited) - 2}"
        files_str = _truncate(files_str, 28)
        t.append(f"  edited: {files_str}", style="dim yellow")
    if session.awaiting_user_input:
        t.append("  awaiting input", style="bold yellow")

    return t


_SUBAGENT_ICONS: dict[str, str] = {
    "explore": "🔍",
    "generalPurpose": "🔧",
    "browser-use": "🌐",
    "default": "▸",
}


def _format_subagent_rows(session: ComposerState) -> list[Text]:
    """Format subagent tree lines for a session."""
    if not session.subagents:
        return []

    rows: list[Text] = []
    total = len(session.subagents)
    for i, sa in enumerate(session.subagents):
        t = Text()
        is_last = i == total - 1
        t.append("└─ " if is_last else "├─ ", style="dim bright_black")
        sa_type = sa.get("type", "default")
        icon = _SUBAGENT_ICONS.get(sa_type, "▸")
        t.append(f"{icon} ", style="bright_magenta")
        t.append(sa_type, style="bold bright_magenta")
        desc = sa.get("description", "")
        if desc:
            t.append(f"  {desc}", style="dim white")
        rows.append(t)
    return rows


def build_dashboard(
    project_dir: Path, start_time: float, *, show_paths: bool = False
) -> Table:
    """Build the full dashboard grid."""
    sessions = get_active_sessions(project_dir)
    composer = sessions[0] if sessions else get_composer_state(project_dir)
    model_str = get_model(project_dir)
    n_rules = count_rules_files(project_dir)
    n_mcps = count_mcps(project_dir)
    n_notepads = count_notepads(project_dir)
    elapsed = session_elapsed(start_time)
    is_running, mem_str = cursor_process_info()

    config_row = Text()
    pills = [
        _pill("📏", n_rules, "Rules", "bright_cyan"),
        Text("  "),
        _pill("🔌", n_mcps, "MCPs", "bright_yellow"),
        Text("  "),
        _pill("📓", n_notepads, "Notepads", "bright_magenta"),
    ]
    for p in pills:
        config_row.append_text(p)

    session_text = Text()
    session_text.append("⏱  ", style="bright_white")
    session_text.append(elapsed, style="bold bright_white")
    if is_running:
        session_text.append(f"   Cursor running  {mem_str}", style="dim white")
    else:
        session_text.append("   Cursor not detected", style="dim red")

    project_text = Text()
    project_text.append(str(project_dir), style="dim white")
    if show_paths:
        sp = cursor_settings_path()
        if sp:
            project_text.append(f"   settings → {sp}", style="dim bright_black")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(min_width=9, style="bold bright_black")
    grid.add_column()

    max_context_pct: float | None = None
    max_context_source: str | None = None

    awaiting_agents: list[str] = []
    if sessions:
        for i, s in enumerate(sessions):
            if len(sessions) > 1:
                row_label = "  AGENTS" if i == 0 else ""
            else:
                row_label = "  AGENT"
            sid = s.composer_id or f"{s.name}:{s.model}:{i}"
            history_key = f"{project_dir}::{sid}"
            grid.add_row(row_label, _format_session_row(s, i, history_key=history_key))
            if s.context_limit > 0:
                pct = (s.context_used / s.context_limit) * 100
                if max_context_pct is None or pct > max_context_pct:
                    max_context_pct = pct
                    max_context_source = f"#{i + 1}"
            if s.awaiting_user_input:
                awaiting_agents.append(f"#{i + 1}")
            for sub_row in _format_subagent_rows(s):
                grid.add_row("", sub_row)
    else:
        label, model_color = classify_model(model_str)
        model_text = Text()
        model_text.append(label, style=f"bold {model_color}")
        model_text.append("   ", style="dim white")
        model_text.append(model_str, style="dim white")
        grid.add_row("  MODEL", model_text)
        grid.add_row("  MODE", _format_mode(composer))
        if composer.context_limit > 0:
            ctx_text = _context_bar(composer.context_used, composer.context_limit)
            pct = (composer.context_used / composer.context_limit) * 100
            _append_trend_hint(ctx_text, f"{project_dir}::single", pct)
            max_context_pct = pct
            max_context_source = None
            grid.add_row("  CONTEXT", ctx_text)
        else:
            used_tok, max_tok = estimate_context_tokens(project_dir, model_str)
            ctx_text = _context_bar(used_tok, max_tok)
            pct = (used_tok / max_tok) * 100 if max_tok else 0.0
            _append_trend_hint(ctx_text, f"{project_dir}::single", pct)
            max_context_pct = pct
            max_context_source = None
            grid.add_row("  CONTEXT", ctx_text)

    context_action = _quick_action_for_context(max_context_pct, max_context_source)
    action_row = _action_todo_row(awaiting_agents, context_action)
    if action_row:
        grid.add_row("  ACTION", action_row)

    grid.add_row("  CONFIG", config_row)
    grid.add_row("  SESSION", session_text)
    grid.add_row("  PROJECT", project_text)

    return grid


def build_all_dashboard(start_time: float) -> Table:
    """Build a dashboard showing sessions across all Cursor workspaces."""
    all_sessions = get_all_active_sessions()
    elapsed = session_elapsed(start_time)
    is_running, mem_str = cursor_process_info()

    grid = Table.grid(padding=(0, 2))
    grid.add_column(min_width=10, style="bold bright_black")
    grid.add_column()

    agent_num = 0
    max_context_pct: float | None = None
    max_context_source: str | None = None
    awaiting_sources: list[str] = []
    if all_sessions:
        for project_name, sessions in sorted(all_sessions.items()):
            # Project header
            proj_text = Text()
            proj_text.append(project_name, style="bold bright_white")
            proj_text.append(
                f"  ({len(sessions)} agent{'s' if len(sessions) != 1 else ''})",
                style="dim white",
            )
            grid.add_row("  PROJECT", proj_text)

            for s in sessions:
                agent_num += 1
                sid = s.composer_id or f"{s.name}:{s.model}:{agent_num - 1}"
                history_key = f"{project_name}::{sid}"
                grid.add_row(
                    "", _format_session_row(s, agent_num - 1, history_key=history_key)
                )
                if s.context_limit > 0:
                    pct = (s.context_used / s.context_limit) * 100
                    if max_context_pct is None or pct > max_context_pct:
                        max_context_pct = pct
                        max_context_source = f"{project_name} #{agent_num}"
                if s.awaiting_user_input:
                    awaiting_sources.append(f"{project_name} #{agent_num}")
                for sub_row in _format_subagent_rows(s):
                    grid.add_row("", sub_row)
    else:
        empty = Text("No active Cursor sessions found", style="dim white")
        grid.add_row("", empty)

    # Session / process row
    session_text = Text()
    session_text.append("⏱  ", style="bright_white")
    session_text.append(elapsed, style="bold bright_white")
    if is_running:
        session_text.append(f"   Cursor running  {mem_str}", style="dim white")
    else:
        session_text.append("   Cursor not detected", style="dim red")

    total = sum(len(s) for s in all_sessions.values())
    summary = Text()
    summary.append(
        f"{len(all_sessions)} project{'s' if len(all_sessions) != 1 else ''}",
        style="bright_cyan",
    )
    summary.append(f"  {total} agent{'s' if total != 1 else ''}", style="bright_green")

    context_action = _quick_action_for_context(max_context_pct, max_context_source)
    action_row = _action_todo_row(awaiting_sources, context_action)
    if action_row:
        grid.add_row("  ACTION", action_row)
    grid.add_row("  TOTAL", summary)
    grid.add_row("  SESSION", session_text)

    return grid


def make_panel(
    project_dir: Path,
    start_time: float,
    *,
    show_all: bool = False,
    show_paths: bool = False,
) -> Panel:
    """Wrap the dashboard grid in a styled panel."""
    title = Text()
    title.append("⊙ ", style="bold bright_cyan")
    title.append("cursor-hub", style="bold white")
    if show_all:
        title.append("  ALL", style="bold bright_yellow")

    subtitle = Text(datetime.now().strftime("%H:%M:%S"), style="dim white")

    if show_all:
        content = build_all_dashboard(start_time)
    else:
        content = build_dashboard(project_dir, start_time, show_paths=show_paths)

    return Panel(
        Align.left(content),
        title=title,
        subtitle=subtitle,
        border_style="bright_black",
        padding=(0, 1),
        box=box.ROUNDED,
    )
