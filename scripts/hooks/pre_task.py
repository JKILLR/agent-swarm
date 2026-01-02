#!/usr/bin/env python3
"""Pre-task hook for agent coordination.

This hook runs before any Task tool invocation to:
1. Log the task to the coordination database
2. Check for conflicts with running tasks
3. Allow or warn about concurrent execution
"""

import json
import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

# Use environment variable or fallback to relative path
SWARM_ROOT = Path(os.environ.get("SWARM_ROOT", Path(__file__).parent.parent.parent))
DB_PATH = SWARM_ROOT / ".claude" / "coordination.db"


def init_db():
    """Initialize coordination database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_log (
            id INTEGER PRIMARY KEY,
            agent TEXT,
            prompt TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            agent TEXT,
            timestamp TEXT,
            UNIQUE(namespace, key)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_agent ON task_log(agent, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ns ON decisions(namespace)")
    conn.commit()
    return conn


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        # Allow on parse error
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    agent_name = tool_input.get("agent", "unknown")
    prompt = tool_input.get("prompt", "")[:500]

    conn = init_db()

    # Log the task as starting
    conn.execute(
        "INSERT INTO task_log (agent, prompt, started_at, status) VALUES (?, ?, ?, ?)",
        (agent_name, prompt, datetime.now().isoformat(), "starting")
    )
    conn.commit()

    # Check for conflicts with already running tasks
    running = conn.execute(
        "SELECT id, agent, prompt FROM task_log WHERE status = 'running' AND agent = ?",
        (agent_name,)
    ).fetchall()

    if running:
        # Warn but allow - parallel execution is intentional
        print(json.dumps({
            "message": f"Note: {agent_name} already has {len(running)} running task(s)",
            "continue": True  # Allow anyway
        }))

    # Update status to running
    conn.execute(
        "UPDATE task_log SET status = 'running' WHERE agent = ? AND status = 'starting'",
        (agent_name,)
    )
    conn.commit()
    conn.close()

    # Exit 0 to allow the task
    sys.exit(0)


if __name__ == "__main__":
    main()
