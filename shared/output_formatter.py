"""Rich output formatting for agent swarm CLI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Agent type color schemes
AGENT_COLORS: dict[str, str] = {
    "supreme": "bold blue",
    "orchestrator": "blue",
    "researcher": "green",
    "implementer": "yellow",
    "critic": "red",
    "benchmarker": "magenta",
    "monitor": "cyan",
    "worker": "white",
}

AGENT_ICONS: dict[str, str] = {
    "supreme": "👑",
    "orchestrator": "🎯",
    "researcher": "🔬",
    "implementer": "🔧",
    "critic": "🔍",
    "benchmarker": "📊",
    "monitor": "👁",
    "worker": "⚙",
}


@dataclass
class AgentMessage:
    """Structured message from an agent."""

    agent_name: str
    agent_type: str
    content: str
    status: str = "complete"  # working, complete, error
    is_subagent: bool = False


class OutputFormatter:
    """Format agent outputs for rich CLI display."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._current_agents: dict[str, str] = {}  # agent_name -> status

    def get_agent_color(self, agent_type: str) -> str:
        """Get color for agent type."""
        return AGENT_COLORS.get(agent_type.lower(), "white")

    def get_agent_icon(self, agent_type: str) -> str:
        """Get icon for agent type."""
        return AGENT_ICONS.get(agent_type.lower(), "•")

    def format_agent_header(self, name: str, agent_type: str, status: str = "complete") -> str:
        """Format agent header with icon and status."""
        icon = self.get_agent_icon(agent_type)
        status_indicator = {
            "working": " (working...)",
            "complete": "",
            "error": " (error)",
        }.get(status, "")
        return f"{icon} {name}{status_indicator}"

    def print_agent_panel(
        self,
        message: AgentMessage,
        width: int | None = None,
    ) -> None:
        """Print a formatted panel for an agent's output."""
        color = self.get_agent_color(message.agent_type)
        header = self.format_agent_header(message.agent_name, message.agent_type, message.status)

        # Parse and clean content
        content = self._clean_content(message.content)

        # Use markdown rendering for content
        try:
            rendered = Markdown(content)
        except Exception:
            rendered = Text(content)

        panel = Panel(
            rendered,
            title=f"[{color}]{header}[/{color}]",
            border_style=color,
            width=width,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_thinking(self, agent_name: str, agent_type: str) -> None:
        """Print a thinking indicator for an agent."""
        color = self.get_agent_color(agent_type)
        icon = self.get_agent_icon(agent_type)
        self.console.print(f"[{color}]{icon} {agent_name}[/{color}] [dim]thinking...[/dim]")

    def print_summary(
        self,
        title: str,
        findings: list[str],
        recommendations: list[str] | None = None,
    ) -> None:
        """Print a formatted summary panel."""
        content_parts = []

        if findings:
            content_parts.append("[bold]Key Findings:[/bold]")
            for finding in findings:
                content_parts.append(f"  • {finding}")

        if recommendations:
            content_parts.append("")
            content_parts.append("[bold]Recommendations:[/bold]")
            for rec in recommendations:
                content_parts.append(f"  → {rec}")

        panel = Panel(
            "\n".join(content_parts),
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_divider(self, label: str | None = None) -> None:
        """Print a visual divider."""
        if label:
            self.console.rule(f"[dim]{label}[/dim]", style="dim")
        else:
            self.console.rule(style="dim")

    def _clean_content(self, content: str) -> str:
        """Clean and format raw content."""
        # Remove SDK internal message formatting
        content = re.sub(r"SystemMessage\(subtype='[^']*'.*?\)", "", content)
        content = re.sub(r"TextBlock\(citations=.*?text='", "", content)
        content = re.sub(r"'\s*,\s*type='text'\)", "", content)
        content = re.sub(r"ToolUseBlock\(.*?\)", "[Tool Use]", content)
        content = re.sub(r"ToolResultBlock\(.*?\)", "", content)

        # Clean up escape sequences
        content = content.replace("\\n", "\n")
        content = content.replace("\\'", "'")
        content = content.replace('\\"', '"')

        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def parse_raw_output(self, raw_output: str) -> list[AgentMessage]:
        """Parse raw SDK output into structured messages."""
        messages = []

        # Try to split by agent sections
        sections = re.split(r"(#{1,3}\s*(?:Researcher|Implementer|Critic|Summary|Recommendation))", raw_output)

        if len(sections) > 1:
            current_agent = "Supreme"
            current_type = "orchestrator"

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Check if this is a header
                header_match = re.match(r"#{1,3}\s*(Researcher|Implementer|Critic|Summary|Recommendation)", section)
                if header_match:
                    current_agent = header_match.group(1)
                    current_type = current_agent.lower()
                    if current_type in ("summary", "recommendation"):
                        current_type = "orchestrator"
                else:
                    messages.append(
                        AgentMessage(
                            agent_name=current_agent,
                            agent_type=current_type,
                            content=section,
                        )
                    )
        else:
            # Single message
            messages.append(
                AgentMessage(
                    agent_name="Supreme Orchestrator",
                    agent_type="orchestrator",
                    content=raw_output,
                )
            )

        return messages

    def format_response(self, raw_output: str) -> None:
        """Format and print a complete response."""
        # Clean the raw output first
        cleaned = self._clean_content(raw_output)

        # Check if this contains multiple agent outputs
        if "Researcher" in cleaned or "Implementer" in cleaned or "Critic" in cleaned:
            # Multi-agent output - try to parse and format separately
            self._format_multi_agent_output(cleaned)
        else:
            # Single agent output
            self.print_agent_panel(
                AgentMessage(
                    agent_name="Supreme Orchestrator",
                    agent_type="orchestrator",
                    content=cleaned,
                )
            )

    def _format_multi_agent_output(self, content: str) -> None:
        """Format output containing multiple agent responses."""
        # Split by major sections
        sections = re.split(
            r"\n(?=#{1,3}\s+|\*\*(?:Researcher|Implementer|Critic|Summary|Key Finding|Recommendation))", content
        )

        current_type = "orchestrator"
        buffer = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Detect section type
            if re.match(r"(?:#{1,3}\s+)?\*?Researcher", section, re.I):
                if buffer:
                    self._print_buffered(current_type, buffer)
                    buffer = []
                current_type = "researcher"
            elif re.match(r"(?:#{1,3}\s+)?\*?Implementer", section, re.I):
                if buffer:
                    self._print_buffered(current_type, buffer)
                    buffer = []
                current_type = "implementer"
            elif re.match(r"(?:#{1,3}\s+)?\*?Critic", section, re.I):
                if buffer:
                    self._print_buffered(current_type, buffer)
                    buffer = []
                current_type = "critic"
            elif re.match(r"(?:#{1,3}\s+)?\*?(?:Summary|Final|Recommendation|Key Finding)", section, re.I):
                if buffer:
                    self._print_buffered(current_type, buffer)
                    buffer = []
                current_type = "summary"

            buffer.append(section)

        # Print remaining buffer
        if buffer:
            self._print_buffered(current_type, buffer)

    def _print_buffered(self, agent_type: str, buffer: list[str]) -> None:
        """Print buffered content for an agent type."""
        content = "\n\n".join(buffer)

        name_map = {
            "researcher": "Researcher",
            "implementer": "Implementer",
            "critic": "Critic",
            "summary": "Summary",
            "orchestrator": "Supreme Orchestrator",
        }

        # For summary, use special formatting
        if agent_type == "summary":
            self.print_agent_panel(
                AgentMessage(
                    agent_name="Summary",
                    agent_type="orchestrator",
                    content=content,
                )
            )
        else:
            self.print_agent_panel(
                AgentMessage(
                    agent_name=name_map.get(agent_type, agent_type.title()),
                    agent_type=agent_type,
                    content=content,
                )
            )


def create_formatter(console: Console | None = None) -> OutputFormatter:
    """Create a new output formatter."""
    return OutputFormatter(console)
