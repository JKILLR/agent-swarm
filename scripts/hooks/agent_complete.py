#!/usr/bin/env python3
"""Agent completion hook for coordination.

This hook runs when a subagent stops to:
1. Update task status in the coordination database
2. Log completion time
3. Enable other agents to see what was completed
"""

import json
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / ".claude" / "coordination.db"


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)

    agent_name = hook_input.get("agent", "unknown")
    result = hook_input.get("result", "")[:1000]

    try:
        conn = sqlite3.connect(str(DB_PATH))

        # Update the most recent running task for this agent
        conn.execute("""
            UPDATE task_log
            SET status = 'completed', completed_at = ?
            WHERE agent = ? AND status = 'running'
            ORDER BY started_at DESC
            LIMIT 1
        """, (datetime.now().isoformat(), agent_name))

        conn.commit()
        conn.close()
    except Exception as e:
        # Log error but don't block
        print(json.dumps({"warning": f"Failed to update task log: {e}"}), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
