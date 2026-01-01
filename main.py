#!/usr/bin/env python3
"""CLI entry point for the Agent Swarm system."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from supreme.orchestrator import SupremeOrchestrator
from shared.swarm_interface import load_swarm
from shared.output_formatter import OutputFormatter, create_formatter

console = Console()
formatter = create_formatter(console)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def get_orchestrator() -> SupremeOrchestrator:
    """Get or create the Supreme Orchestrator."""
    return SupremeOrchestrator(
        base_path=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config.yaml",
        logs_dir=PROJECT_ROOT / "logs",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """Agent Swarm - Hierarchical AI Agent Management System.

    Manage multiple AI-assisted projects through a Supreme Orchestrator
    that coordinates specialized swarms of agents with parallel execution.
    """
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


@cli.command("list")
def list_swarms() -> None:
    """List all swarms and their status."""
    orchestrator = get_orchestrator()

    if not orchestrator.swarms:
        console.print("[yellow]No swarms found.[/yellow]")
        console.print("Create one with: python main.py new <name>")
        return

    table = Table(title="Agent Swarms", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Agents", justify="right")
    table.add_column("Description")
    table.add_column("Top Priority")

    for swarm in orchestrator.swarms.values():
        status = swarm.get_status()
        status_style = {
            "active": "[green]active[/green]",
            "paused": "[yellow]paused[/yellow]",
            "archived": "[dim]archived[/dim]",
        }.get(status["status"], status["status"])

        priorities = status.get("priorities", [])
        top_priority = priorities[0] if priorities else "-"

        table.add_row(
            status["name"],
            status_style,
            str(status["agent_count"]),
            status["description"][:40] + "..." if len(status["description"]) > 40 else status["description"],
            top_priority[:30] + "..." if len(top_priority) > 30 else top_priority,
        )

    console.print(table)

    # Show total agents
    console.print(f"\n[dim]Total agents across all swarms: {len(orchestrator.all_agents)}[/dim]")


@cli.command("status")
@click.argument("swarm", required=False)
def show_status(swarm: Optional[str]) -> None:
    """Show detailed status of a swarm or all swarms."""
    orchestrator = get_orchestrator()

    if swarm:
        # Show specific swarm status
        swarm_obj = orchestrator.get_swarm(swarm)
        if not swarm_obj:
            console.print(f"[red]Swarm '{swarm}' not found.[/red]")
            return

        status = swarm_obj.get_status()

        # Create detailed panel
        details = []
        details.append(f"**Name:** {status['name']}")
        details.append(f"**Version:** {status['version']}")
        details.append(f"**Status:** {status['status']}")
        details.append(f"**Description:** {status['description']}")
        details.append(f"**Workspace:** {status['workspace']}")
        details.append("")
        details.append("**Agents:**")
        for agent in status["agents"]:
            agent_type = agent.get('type', agent.get('role', 'worker'))
            bg_marker = " [bg]" if agent.get('background') else ""
            model = agent.get('model', 'sonnet')
            details.append(f"  - {agent['name']} ({agent_type}){bg_marker} - {model}")
        details.append("")
        details.append("**Priorities:**")
        for i, priority in enumerate(status.get("priorities", []), 1):
            details.append(f"  {i}. {priority}")

        console.print(Panel(
            Markdown("\n".join(details)),
            title=f"[bold]{status['name']}[/bold]",
            border_style="cyan",
        ))

    else:
        # Show all swarms overview
        all_status = orchestrator.get_all_status()

        console.print(f"\n[bold]Total Swarms:[/bold] {all_status['total_swarms']}")
        console.print(f"[bold]Total Agents:[/bold] {all_status.get('total_agents', 0)}\n")

        for name, status in all_status["swarms"].items():
            status_color = {
                "active": "green",
                "paused": "yellow",
                "archived": "dim",
            }.get(status["status"], "white")

            # Format agents with types
            agent_list = []
            for agent in status.get("agents", []):
                agent_type = agent.get('type', agent.get('role', 'worker'))
                bg = "*" if agent.get('background') else ""
                agent_list.append(f"{agent['name']}{bg}")

            console.print(Panel(
                f"Status: [{status_color}]{status['status']}[/{status_color}]\n"
                f"Agents: {', '.join(agent_list)}\n"
                f"Tasks: {status.get('current_tasks', 0)}",
                title=f"[bold]{name}[/bold]",
                border_style=status_color,
                width=60,
            ))


@cli.command("new")
@click.argument("name")
@click.option("--description", "-d", default="", help="Swarm description")
@click.option("--template", "-t", default="_template", help="Template to use")
def create_swarm(name: str, description: str, template: str) -> None:
    """Create a new swarm from template."""
    orchestrator = get_orchestrator()

    try:
        swarm = orchestrator.create_swarm(name, description, template)
        console.print(f"[green]✓[/green] Created swarm: [bold]{swarm.name}[/bold]")
        console.print(f"  Location: {swarm.swarm_path}")
        console.print(f"  Agents: {', '.join(a.name for a in swarm.agents.values())}")
        console.print("\nNext steps:")
        console.print("  1. Edit swarm.yaml to customize settings")
        console.print("  2. Update agent prompts in agents/")
        console.print("  3. Run: python main.py status " + name.lower().replace(" ", "_"))
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command("chat")
def interactive_chat() -> None:
    """Start interactive chat with the Supreme Orchestrator."""
    orchestrator = get_orchestrator()

    console.print(Panel(
        "[bold]Agent Swarm - Interactive Mode[/bold]\n\n"
        "Chat with the Supreme Orchestrator to manage your swarms.\n"
        "The orchestrator can spawn parallel agents for complex tasks.\n\n"
        "Type 'quit' or 'exit' to leave.\n"
        "Type 'help' for available commands.",
        border_style="cyan",
    ))

    # Show available swarms
    if orchestrator.swarms:
        console.print("\n[dim]Available swarms:[/dim]")
        for name, swarm in orchestrator.swarms.items():
            status = swarm.config.status
            agent_count = len(swarm.agents)
            console.print(f"  • {name} [{status}] - {agent_count} agents")
    else:
        console.print("\n[yellow]No swarms found. Create one with 'new <name>'[/yellow]")

    console.print()

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "help":
            console.print(Panel(
                "**Interactive Commands:**\n"
                "  • `list` - List all swarms\n"
                "  • `status <swarm>` - Show swarm status\n"
                "  • `new <name>` - Create new swarm\n"
                "  • `quit` / `exit` - Exit chat\n\n"
                "**Or ask anything:**\n"
                "  • Route requests to swarms\n"
                "  • Get project overviews\n"
                "  • Coordinate cross-swarm activities\n"
                "  • Run parallel agent tasks",
                title="Help",
                border_style="green",
            ))
            continue

        # Handle quick commands
        if user_input.lower() == "list":
            list_swarms.callback()
            continue

        if user_input.lower().startswith("status"):
            parts = user_input.split(maxsplit=1)
            swarm_name = parts[1] if len(parts) > 1 else None
            show_status.callback(swarm_name)
            continue

        if user_input.lower().startswith("new "):
            name = user_input[4:].strip()
            if name:
                create_swarm.callback(name, "", "_template")
            continue

        # Route through Supreme Orchestrator
        formatter.print_thinking("Supreme Orchestrator", "orchestrator")

        try:
            response = asyncio.run(orchestrator.route_request(user_input))
            formatter.print_divider()
            formatter.format_response(response)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@cli.command("run")
@click.argument("swarm")
@click.argument("directive")
@click.option("--parallel", "-p", is_flag=True, help="Run with parallel agents")
def run_directive(swarm: str, directive: str, parallel: bool) -> None:
    """Send a directive to a specific swarm.

    Use --parallel to spawn researcher, implementer, and critic in parallel.
    """
    orchestrator = get_orchestrator()

    swarm_obj = orchestrator.get_swarm(swarm)
    if not swarm_obj:
        console.print(f"[red]Swarm '{swarm}' not found.[/red]")
        return

    if parallel:
        console.print(f"[bold]Running on {swarm} with parallel agents...[/bold]\n")

        async def run_parallel():
            collected_output = []
            async for message in orchestrator.run_parallel_on_swarm(swarm, directive):
                if isinstance(message, dict):
                    agent_name = message.get("agent", "Agent")
                    agent_type = message.get("type", "worker")
                    content = message.get("content", message.get("result", str(message)))
                    formatter.print_thinking(agent_name, agent_type)
                else:
                    content = str(message)
                collected_output.append(content)

            return "\n".join(collected_output)

        try:
            result = asyncio.run(run_parallel())
            formatter.print_divider("Results")
            formatter.format_response(result)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
    else:
        formatter.print_thinking(swarm, "orchestrator")

        try:
            response = asyncio.run(orchestrator.send_directive(swarm, directive))
            formatter.print_divider()
            formatter.format_response(response)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@cli.command("agents")
@click.argument("swarm", required=False)
def list_agents(swarm: Optional[str]) -> None:
    """List all agents, optionally filtered by swarm."""
    orchestrator = get_orchestrator()

    table = Table(title="Available Agents", show_header=True, header_style="bold cyan")
    table.add_column("Agent", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Model")
    table.add_column("Background", justify="center")
    table.add_column("Description")

    if swarm:
        swarm_obj = orchestrator.get_swarm(swarm)
        if not swarm_obj:
            console.print(f"[red]Swarm '{swarm}' not found.[/red]")
            return

        for name, defn in swarm_obj.agent_definitions.items():
            table.add_row(
                name,
                defn.agent_type,
                defn.model,
                "✓" if defn.background else "",
                defn.description[:50] + "..." if len(defn.description) > 50 else defn.description,
            )
    else:
        for name, defn in orchestrator.all_agents.items():
            table.add_row(
                name,
                defn.agent_type,
                defn.model,
                "✓" if defn.background else "",
                defn.description[:50] + "..." if len(defn.description) > 50 else defn.description,
            )

    console.print(table)


@cli.command("pause")
@click.argument("swarm")
def pause_swarm(swarm: str) -> None:
    """Pause a swarm."""
    orchestrator = get_orchestrator()

    if orchestrator.pause_swarm(swarm):
        console.print(f"[green]✓[/green] Paused swarm: {swarm}")
    else:
        console.print(f"[red]Swarm '{swarm}' not found.[/red]")


@cli.command("activate")
@click.argument("swarm")
def activate_swarm(swarm: str) -> None:
    """Activate a paused swarm."""
    orchestrator = get_orchestrator()

    if orchestrator.activate_swarm(swarm):
        console.print(f"[green]✓[/green] Activated swarm: {swarm}")
    else:
        console.print(f"[red]Swarm '{swarm}' not found.[/red]")


@cli.command("archive")
@click.argument("swarm")
def archive_swarm(swarm: str) -> None:
    """Archive a swarm."""
    orchestrator = get_orchestrator()

    if orchestrator.archive_swarm(swarm):
        console.print(f"[green]✓[/green] Archived swarm: {swarm}")
    else:
        console.print(f"[red]Swarm '{swarm}' not found.[/red]")


if __name__ == "__main__":
    cli()
