# Agent Swarm

A hierarchical agent system for managing multiple AI-assisted projects.

## Overview

Agent Swarm provides a meta-system where a **Supreme Orchestrator** manages multiple project "swarms" (research projects, applications, trading bots, etc.). Each swarm has its own agents that work together with consensus-based decision making.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Supreme Orchestrator                       │
│         (Routes requests, monitors all swarms)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Swarm A │   │ Swarm B │   │ Swarm C │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
   │Orchestr.│   │Orchestr.│   │Orchestr.│
   │ Worker  │   │ Worker  │   │ Worker  │
   │ Critic  │   │ Critic  │   │ Critic  │
   └─────────┘   └─────────┘   └─────────┘
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **List available swarms:**
   ```bash
   python main.py list
   ```

3. **Create a new swarm:**
   ```bash
   python main.py new "My Project" --description "A new AI project"
   ```

4. **Start interactive chat:**
   ```bash
   python main.py chat
   ```

5. **Send a directive to a swarm:**
   ```bash
   python main.py run myproject "Analyze the current codebase and suggest improvements"
   ```

## Creating New Swarms

New swarms are created by copying from the `_template` directory:

```bash
python main.py new "Trading Bot" -d "Algorithmic trading system"
```

This creates a new swarm with:
- `swarm.yaml` - Configuration file
- `agents/` - Agent system prompts
  - `orchestrator.md` - Swarm coordinator
  - `worker.md` - Implementation agent
  - `critic.md` - Adversarial reviewer
- `workspace/` - Swarm's working directory

### Customizing Your Swarm

1. Edit `swarms/<your-swarm>/swarm.yaml` to set priorities and settings
2. Customize agent prompts in `agents/*.md` for your domain
3. Add any project files to `workspace/`

## CLI Commands

| Command | Description |
|---------|-------------|
| `list` | List all swarms and their status |
| `status [swarm]` | Show detailed status |
| `new <name>` | Create a new swarm from template |
| `chat` | Interactive mode with Supreme Orchestrator |
| `run <swarm> <directive>` | Send directive to specific swarm |
| `pause <swarm>` | Pause a swarm |
| `activate <swarm>` | Activate a paused swarm |
| `archive <swarm>` | Archive a swarm |

## Consensus Protocol

For major decisions, swarms use consensus voting:

- **APPROVE** - Full support
- **APPROVE_WITH_CHANGES** - Support with modifications
- **REQUEST_MORE_INFO** - Need clarification
- **REJECT** - Oppose (blocks the proposal)

Any REJECT vote blocks the proposal. The critic agent is designed to challenge proposals and ensure quality.

## Configuration

Edit `config.yaml` to customize:
- Default models
- Agent tool permissions
- Consensus requirements
- Logging settings

## Directory Structure

```
agent-swarm/
├── config.yaml           # System configuration
├── main.py               # CLI entry point
├── shared/               # Shared components
│   ├── agent_base.py     # Base agent wrapper
│   ├── swarm_interface.py # Swarm management
│   └── consensus.py      # Consensus protocol
├── supreme/              # Supreme Orchestrator
│   └── orchestrator.py
├── swarms/               # Individual swarms
│   └── _template/        # Template for new swarms
└── logs/                 # Agent conversation logs
```

## Development

The system is designed to be extensible:

- **Add new agent roles**: Create new prompt files in `agents/`
- **Custom consensus logic**: Extend `ConsensusProtocol`
- **New swarm types**: Create specialized templates
- **Tool integrations**: Extend `BaseAgent` tools

## License

MIT License
