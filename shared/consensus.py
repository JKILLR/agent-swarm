"""Consensus protocol for agent decision making."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_base import BaseAgent

logger = logging.getLogger(__name__)


class Vote(Enum):
    """Vote options for consensus rounds."""

    APPROVE = "approve"
    APPROVE_WITH_CHANGES = "approve_with_changes"
    REQUEST_MORE_INFO = "request_more_info"
    REJECT = "reject"

    @classmethod
    def from_string(cls, value: str) -> Vote:
        """Parse vote from string, case-insensitive."""
        value_lower = value.lower().strip()

        # Handle variations
        if value_lower in ("approve", "approved", "yes", "accept", "agree"):
            return cls.APPROVE
        elif value_lower in ("approve_with_changes", "approve with changes", "conditional", "conditionally approve"):
            return cls.APPROVE_WITH_CHANGES
        elif value_lower in ("request_more_info", "request more info", "need more info", "more info", "unclear"):
            return cls.REQUEST_MORE_INFO
        elif value_lower in ("reject", "rejected", "no", "deny", "disagree", "oppose"):
            return cls.REJECT
        else:
            # Default to request more info if unclear
            logger.warning(f"Unknown vote value '{value}', defaulting to REQUEST_MORE_INFO")
            return cls.REQUEST_MORE_INFO


@dataclass
class ConsensusResult:
    """Result of a consensus round."""

    approved: bool
    outcome: str  # approved, approved_with_changes, rejected, needs_more_info
    votes: dict[str, Vote]
    discussion: list[dict[str, str]]
    proposal: str
    changes_requested: list[str] = field(default_factory=list)
    info_requests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "outcome": self.outcome,
            "votes": {k: v.value for k, v in self.votes.items()},
            "discussion": self.discussion,
            "proposal": self.proposal,
            "changes_requested": self.changes_requested,
            "info_requests": self.info_requests,
        }


@dataclass
class ConsensusRound:
    """A single consensus round."""

    topic: str
    proposal: str
    proposer: str
    votes: dict[str, Vote] = field(default_factory=dict)
    discussion: list[dict[str, str]] = field(default_factory=list)
    outcome: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "proposal": self.proposal,
            "proposer": self.proposer,
            "votes": {k: v.value for k, v in self.votes.items()},
            "discussion": self.discussion,
            "outcome": self.outcome,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ConsensusProtocol:
    """Protocol for running consensus rounds among agents."""

    VOTE_PROMPT = """You are participating in a consensus vote. Please review the following proposal and provide your vote.

**Topic:** {topic}

**Proposal:**
{proposal}

**Your Role:** As {role}, evaluate this proposal from your perspective.

**Voting Options:**
1. APPROVE - You fully support this proposal
2. APPROVE_WITH_CHANGES - You support it but suggest specific modifications
3. REQUEST_MORE_INFO - You need clarification before voting
4. REJECT - You oppose this proposal (explain why)

**Instructions:**
1. Analyze the proposal thoroughly
2. Consider implications from your role's perspective
3. Provide your reasoning
4. State your vote clearly at the end using the format: **VOTE: [YOUR_VOTE]**

If you vote APPROVE_WITH_CHANGES, list your suggested changes.
If you vote REQUEST_MORE_INFO, specify what information you need.
If you vote REJECT, explain your objections clearly.
"""

    def __init__(self, logs_dir: Path | None = None) -> None:
        """Initialize the consensus protocol.

        Args:
            logs_dir: Directory for consensus logs
        """
        self.logs_dir = logs_dir or Path("./logs/consensus")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[ConsensusRound] = []

    def _parse_vote(self, response: str) -> tuple[Vote, str | None]:
        """Parse vote from agent response.

        Args:
            response: Agent's response text

        Returns:
            Tuple of (Vote, optional details)
        """
        # Look for explicit vote pattern
        vote_pattern = r"\*?\*?VOTE:\s*\*?\*?\s*(\w+(?:\s+\w+)*)"
        match = re.search(vote_pattern, response, re.IGNORECASE)

        if match:
            vote_text = match.group(1).strip()
            vote = Vote.from_string(vote_text)
        else:
            # Try to infer from response content
            response_lower = response.lower()
            if "reject" in response_lower or "oppose" in response_lower:
                vote = Vote.REJECT
            elif "approve with changes" in response_lower or "conditional" in response_lower:
                vote = Vote.APPROVE_WITH_CHANGES
            elif "need more info" in response_lower or "unclear" in response_lower:
                vote = Vote.REQUEST_MORE_INFO
            elif "approve" in response_lower or "support" in response_lower:
                vote = Vote.APPROVE
            else:
                vote = Vote.REQUEST_MORE_INFO

        # Extract details based on vote type
        details = None
        if vote == Vote.APPROVE_WITH_CHANGES:
            # Try to find suggested changes
            changes_pattern = r"(?:changes?|modifications?|suggestions?):\s*(.+?)(?:\n\n|\*\*|$)"
            changes_match = re.search(changes_pattern, response, re.IGNORECASE | re.DOTALL)
            if changes_match:
                details = changes_match.group(1).strip()
        elif vote == Vote.REQUEST_MORE_INFO:
            # Try to find info requests
            info_pattern = r"(?:need|require|clarify|question)(?:s)?:\s*(.+?)(?:\n\n|\*\*|$)"
            info_match = re.search(info_pattern, response, re.IGNORECASE | re.DOTALL)
            if info_match:
                details = info_match.group(1).strip()

        return vote, details

    def _determine_outcome(
        self,
        votes: dict[str, Vote],
    ) -> tuple[bool, str]:
        """Determine the outcome of voting.

        Args:
            votes: Dictionary of agent names to their votes

        Returns:
            Tuple of (approved, outcome_string)
        """
        if not votes:
            return False, "no_votes"

        vote_values = list(votes.values())

        # Any REJECT means rejected
        if Vote.REJECT in vote_values:
            return False, "rejected"

        # Any REQUEST_MORE_INFO means needs more info
        if Vote.REQUEST_MORE_INFO in vote_values:
            return False, "needs_more_info"

        # All approved (with or without changes)
        if Vote.APPROVE_WITH_CHANGES in vote_values:
            return True, "approved_with_changes"

        return True, "approved"

    async def start_round(
        self,
        topic: str,
        proposal: str,
        proposer: str = "system",
    ) -> ConsensusRound:
        """Start a new consensus round.

        Args:
            topic: The topic of the consensus
            proposal: The proposal text
            proposer: Who is proposing

        Returns:
            New ConsensusRound instance
        """
        round_obj = ConsensusRound(
            topic=topic,
            proposal=proposal,
            proposer=proposer,
        )
        self.history.append(round_obj)
        logger.info(f"Started consensus round: {topic}")
        return round_obj

    async def gather_vote(
        self,
        round_obj: ConsensusRound,
        agent: BaseAgent,
    ) -> tuple[Vote, str]:
        """Gather vote from a single agent.

        Args:
            round_obj: The consensus round
            agent: The agent to gather vote from

        Returns:
            Tuple of (Vote, response text)
        """
        prompt = self.VOTE_PROMPT.format(
            topic=round_obj.topic,
            proposal=round_obj.proposal,
            role=agent.role,
        )

        response = await agent.run_sync(prompt)
        vote, details = self._parse_vote(response)

        # Record the vote and discussion
        round_obj.votes[agent.name] = vote
        round_obj.discussion.append(
            {
                "agent": agent.name,
                "role": agent.role,
                "response": response,
                "vote": vote.value,
                "details": details,
            }
        )

        logger.debug(f"Agent {agent.name} voted: {vote.value}")
        return vote, response

    async def run_consensus(
        self,
        topic: str,
        proposal: str,
        voters: list[BaseAgent],
        proposer: str = "system",
    ) -> ConsensusResult:
        """Run a complete consensus round.

        Args:
            topic: The topic of the consensus
            proposal: The proposal text
            voters: List of agents who will vote
            proposer: Who is proposing

        Returns:
            ConsensusResult with the outcome
        """
        round_obj = await self.start_round(topic, proposal, proposer)

        # Gather votes from all agents
        changes_requested = []
        info_requests = []

        for agent in voters:
            try:
                vote, response = await self.gather_vote(round_obj, agent)

                # Collect changes and info requests
                vote_entry = round_obj.discussion[-1]
                if vote == Vote.APPROVE_WITH_CHANGES and vote_entry.get("details"):
                    changes_requested.append(f"{agent.name}: {vote_entry['details']}")
                elif vote == Vote.REQUEST_MORE_INFO and vote_entry.get("details"):
                    info_requests.append(f"{agent.name}: {vote_entry['details']}")

            except Exception as e:
                logger.error(f"Error gathering vote from {agent.name}: {e}")
                # Default to request more info on error
                round_obj.votes[agent.name] = Vote.REQUEST_MORE_INFO
                round_obj.discussion.append(
                    {
                        "agent": agent.name,
                        "role": agent.role,
                        "response": f"Error: {e}",
                        "vote": Vote.REQUEST_MORE_INFO.value,
                    }
                )

        # Determine outcome
        approved, outcome = self._determine_outcome(round_obj.votes)
        round_obj.outcome = outcome
        round_obj.completed_at = datetime.now()

        # Save to logs
        self._save_round(round_obj)

        result = ConsensusResult(
            approved=approved,
            outcome=outcome,
            votes=round_obj.votes.copy(),
            discussion=round_obj.discussion.copy(),
            proposal=proposal,
            changes_requested=changes_requested,
            info_requests=info_requests,
        )

        logger.info(f"Consensus round completed: {outcome} (approved={approved})")
        return result

    def _save_round(self, round_obj: ConsensusRound) -> None:
        """Save consensus round to logs."""
        filename = f"consensus_{round_obj.started_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.logs_dir / filename

        with open(filepath, "w") as f:
            json.dump(round_obj.to_dict(), f, indent=2)

        logger.debug(f"Saved consensus round to {filepath}")

    def get_history(self) -> list[ConsensusRound]:
        """Get all consensus rounds."""
        return self.history.copy()

    def get_recent(self, count: int = 10) -> list[ConsensusRound]:
        """Get recent consensus rounds."""
        return self.history[-count:]
