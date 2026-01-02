#!/usr/bin/env python3
"""
Manual tracking synchronization tool.
Use this to manually sync task status when commit hooks aren't available.

Usage:
  python tools/sync_tracking.py <swarm_name> <task_id> <status>

Example:
  python tools/sync_tracking.py swarm_dev P1 completed
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

def sync_task_status(swarm_name: str, task_id: str, new_status: str):
    """Update task status in swarm.yaml file"""

    swarm_yaml_path = Path(__file__).parent.parent / 'swarms' / swarm_name / 'swarm.yaml'

    if not swarm_yaml_path.exists():
        print(f"❌ Error: Swarm '{swarm_name}' not found at {swarm_yaml_path}")
        return False

    try:
        # Load swarm config
        with open(swarm_yaml_path, 'r') as f:
            swarm_config = yaml.safe_load(f)

        # Find and update task
        task_found = False
        if 'priorities' in swarm_config:
            for priority in swarm_config['priorities']:
                # Match by ID or by title (converted to slug)
                priority_id = priority.get('id')
                priority_slug = priority.get('title', '').lower().replace(' ', '-')

                if priority_id == task_id or priority_slug == task_id.lower():
                    old_status = priority.get('status', 'unknown')
                    priority['status'] = new_status
                    priority['last_updated'] = datetime.now().strftime('%Y-%m-%d')

                    # Add note about manual update
                    if 'notes' not in priority:
                        priority['notes'] = []
                    priority['notes'].append(f"Manual status update: {old_status} → {new_status} on {datetime.now().strftime('%Y-%m-%d %H:%M')}")

                    task_found = True

                    # Write updated config
                    with open(swarm_yaml_path, 'w') as f:
                        yaml.dump(swarm_config, f, default_flow_style=False, sort_keys=False)

                    print(f"✅ Updated: {swarm_name}/{task_id}")
                    print(f"   Old status: {old_status}")
                    print(f"   New status: {new_status}")
                    print(f"   File: {swarm_yaml_path}")
                    return True

        if not task_found:
            print(f"❌ Error: Task '{task_id}' not found in {swarm_name}")
            print(f"\nAvailable tasks:")
            if 'priorities' in swarm_config:
                for p in swarm_config['priorities']:
                    print(f"  - {p.get('id', 'NO_ID')}: {p.get('title', 'NO_TITLE')} (status: {p.get('status', 'unknown')})")
            return False

    except Exception as e:
        print(f"❌ Error updating swarm status: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python tools/sync_tracking.py <swarm_name> <task_id> <status>")
        print("\nExample:")
        print("  python tools/sync_tracking.py swarm_dev P1 completed")
        print("\nValid statuses: not_started, in_progress, completed, blocked, on_hold, cancelled")
        sys.exit(1)

    swarm_name = sys.argv[1]
    task_id = sys.argv[2]
    new_status = sys.argv[3]

    # Validate status
    valid_statuses = ['not_started', 'in_progress', 'completed', 'blocked', 'on_hold', 'cancelled']
    if new_status not in valid_statuses:
        print(f"❌ Error: Invalid status '{new_status}'")
        print(f"Valid statuses: {', '.join(valid_statuses)}")
        sys.exit(1)

    success = sync_task_status(swarm_name, task_id, new_status)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
