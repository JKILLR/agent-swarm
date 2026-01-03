"""Shared components for the agent swarm system."""

from .agent_base import AgentConfig, BaseAgent
from .agent_definitions import (
    AGENT_TYPES,
    AgentDefinition,
    get_agent_type,
    list_agent_types,
    load_agent,
    load_agent_from_file,
    parse_frontmatter,
)
from .consensus import ConsensusProtocol, ConsensusResult, ConsensusRound, Vote
from .swarm_interface import Swarm, SwarmConfig, SwarmInterface, load_swarm

# Work Ledger (Gas Town Beads-style persistence)
from .work_models import (
    WorkHistoryEntry,
    WorkIndex,
    WorkItem,
    WorkPriority,
    WorkStatus,
    WorkType,
)
from .work_ledger import WorkLedger, get_work_ledger

# Agent Mailbox (Gas Town-style structured handoffs)
from .agent_mailbox import (
    HandoffContext,
    MailboxManager,
    Message,
    MessagePriority,
    MessageStatus,
    MessageType,
    broadcast_to_swarm,
    check_my_mailbox,
    get_mailbox_manager,
    send_handoff,
    send_message,
)

# Auto-Spawn (automatic agent spawning on work detection)
from .auto_spawn import (
    enable_auto_spawn,
    disable_auto_spawn,
    should_auto_spawn,
    get_agent_for_work_type,
    spawn_agent_for_work,
)

__all__ = [
    # Agent base
    "AgentConfig",
    "BaseAgent",
    # Agent definitions
    "AgentDefinition",
    "AGENT_TYPES",
    "load_agent",
    "load_agent_from_file",
    "get_agent_type",
    "list_agent_types",
    "parse_frontmatter",
    # Swarm interface
    "SwarmConfig",
    "SwarmInterface",
    "Swarm",
    "load_swarm",
    # Consensus
    "Vote",
    "ConsensusRound",
    "ConsensusProtocol",
    "ConsensusResult",
    # Work Ledger
    "WorkStatus",
    "WorkType",
    "WorkPriority",
    "WorkHistoryEntry",
    "WorkItem",
    "WorkIndex",
    "WorkLedger",
    "get_work_ledger",
    # Agent Mailbox
    "MessageType",
    "MessagePriority",
    "MessageStatus",
    "Message",
    "HandoffContext",
    "MailboxManager",
    "get_mailbox_manager",
    "send_message",
    "check_my_mailbox",
    "send_handoff",
    "broadcast_to_swarm",
    # Auto-Spawn
    "enable_auto_spawn",
    "disable_auto_spawn",
    "should_auto_spawn",
    "get_agent_for_work_type",
    "spawn_agent_for_work",
]
