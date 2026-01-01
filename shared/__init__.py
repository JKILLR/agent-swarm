"""Shared components for the agent swarm system."""

from .agent_base import AgentConfig, BaseAgent
from .agent_definitions import (
    AgentDefinition,
    AGENT_TYPES,
    load_agent,
    load_agent_from_file,
    get_agent_type,
    list_agent_types,
    parse_frontmatter,
)
from .swarm_interface import SwarmConfig, SwarmInterface, Swarm, load_swarm
from .consensus import Vote, ConsensusRound, ConsensusProtocol, ConsensusResult

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
