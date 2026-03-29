"""Config counting — rules, MCPs, and notepads."""

from pathlib import Path

from .paths import read_json


def count_rules_files(project_dir: Path) -> int:
    """
    Count all rule files Cursor loads:
    - .cursorrules  (legacy, root of project)
    - .cursor/rules/*.mdc  (new-style, project)
    - ~/.cursor/rules/*.mdc  (global)
    """
    found: set[Path] = set()

    current = project_dir.resolve()
    while True:
        cr = current / ".cursorrules"
        if cr.exists():
            found.add(cr)
        parent = current.parent
        if parent == current:
            break
        current = parent

    proj_rules = project_dir / ".cursor" / "rules"
    if proj_rules.is_dir():
        found.update(proj_rules.glob("**/*.mdc"))

    global_rules = Path.home() / ".cursor" / "rules"
    if global_rules.is_dir():
        found.update(global_rules.glob("**/*.mdc"))

    return len(found)


def count_mcps(project_dir: Path) -> int:
    """Count MCP servers from ~/.cursor/mcp.json + .cursor/mcp.json (same name once)."""
    merged: dict[str, object] = {}
    for mcp_path in (
        Path.home() / ".cursor" / "mcp.json",
        project_dir / ".cursor" / "mcp.json",
    ):
        data = read_json(mcp_path)
        servers = data.get("mcpServers", {})
        if isinstance(servers, dict):
            merged.update(servers)
        elif isinstance(servers, list):
            for i, _ in enumerate(servers):
                merged[f"__list_{mcp_path}_{i}"] = True
    return len(merged)


def count_notepads(project_dir: Path) -> int:
    """Count Cursor notepads (.cursor/notepads/*.md)."""
    np_dir = project_dir / ".cursor" / "notepads"
    if not np_dir.is_dir():
        return 0
    return len(list(np_dir.glob("*.md")))
