# Git Hooks Setup for Automated Tracking

## Manual Installation

To enable automated task tracking via git commits, install the commit-msg hook:

```bash
# Make the hook executable
chmod +x .git/hooks/commit-msg

# Or create it manually
cat > .git/hooks/commit-msg << 'EOF'
#!/usr/bin/env python3
"""
Git commit-msg hook for automated swarm task tracking.
"""
import sys, re, yaml
from pathlib import Path
from datetime import datetime

commit_msg_file = sys.argv[1]
with open(commit_msg_file, 'r') as f:
    commit_msg = f.read()

swarm_match = re.search(r'\[swarm:(\w+)\]', commit_msg)
task_match = re.search(r'\[task:([\w-]+)\]', commit_msg)
status_match = re.search(r'\[status:(\w+)\]', commit_msg)

if swarm_match and task_match and status_match:
    swarm_name = swarm_match.group(1)
    task_id = task_match.group(1)
    new_status = status_match.group(1)
    swarm_yaml_path = Path(__file__).parent.parent.parent / 'swarms' / swarm_name / 'swarm.yaml'

    if swarm_yaml_path.exists():
        try:
            with open(swarm_yaml_path, 'r') as f:
                swarm_config = yaml.safe_load(f)
            if 'priorities' in swarm_config:
                for priority in swarm_config['priorities']:
                    if priority.get('id') == task_id or priority.get('title', '').lower().replace(' ', '-') == task_id.lower():
                        priority['status'] = new_status
                        priority['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                        with open(swarm_yaml_path, 'w') as f:
                            yaml.dump(swarm_config, f, default_flow_style=False, sort_keys=False)
                        print(f"✅ Auto-updated: {swarm_name}/{task_id} → {new_status}")
                        break
        except Exception as e:
            print(f"⚠️  Warning: Could not auto-update swarm status: {e}", file=sys.stderr)
sys.exit(0)
EOF

chmod +x .git/hooks/commit-msg
```

## Verification

Test the hook is installed:
```bash
ls -la .git/hooks/commit-msg
# Should show: -rwxr-xr-x ... commit-msg
```

## Usage

Commit with tracking tags:
```bash
git commit -m "feat: Complete feature [swarm:swarm_dev] [task:P1] [status:completed]"
```

You should see:
```
✅ Auto-updated: swarm_dev/P1 → completed
```
