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
]
