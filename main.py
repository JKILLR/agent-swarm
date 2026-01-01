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

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from supreme.orchestrator import SupremeOrchestrator
from shared.swarm_interface import load_swarm

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def get_orchestrator() -> SupremeOrchestrator:
    """Get or create the Supreme Orchestrator."""
    config_path = PROJECT_ROOT / "config.yaml"
    swarms_dir = PROJECT_ROOT / "swarms"
    logs_dir = PROJECT_ROOT / "logs"

    return SupremeOrchestrator(
        swarms_dir=swarms_dir,
        config_path=config_path if config_path.exists() else None,
        logs_dir=logs_dir,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """Agent Swarm - Hierarchical AI Agent Management System.

    Manage multiple AI-assisted projects through a Supreme Orchestrator
    that coordinates specialized swarms of agents.
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
            details.append(f"  - {agent['name']} ({agent['role']})")
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

        console.print(f"\n[bold]Total Swarms:[/bold] {all_status['total_swarms']}\n")

        for name, status in all_status["swarms"].items():
            status_color = {
                "active": "green",
                "paused": "yellow",
                "archived": "dim",
            }.get(status["status"], "white")

            console.print(Panel(
                f"Status: [{status_color}]{status['status']}[/{status_color}]\n"
                f"Agents: {status['agent_count']}\n"
                f"Tasks: {status.get('current_tasks', 0)}",
                title=f"[bold]{name}[/bold]",
                border_style=status_color,
                width=50,
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
        "Type 'quit' or 'exit' to leave.\n"
        "Type 'help' for available commands.",
        border_style="cyan",
    ))

    # Show available swarms
    if orchestrator.swarms:
        console.print("\n[dim]Available swarms:[/dim]")
        for name, swarm in orchestrator.swarms.items():
            status = swarm.config.status
            console.print(f"  • {name} [{status}]")
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
                "  • Coordinate cross-swarm activities",
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
        console.print("[dim]Thinking...[/dim]")

        try:
            response = asyncio.run(orchestrator.route_request(user_input))
            console.print(Panel(
                Markdown(response),
                title="[bold]Supreme Orchestrator[/bold]",
                border_style="blue",
            ))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@cli.command("run")
@click.argument("swarm")
@click.argument("directive")
def run_directive(swarm: str, directive: str) -> None:
    """Send a directive to a specific swarm."""
    orchestrator = get_orchestrator()

    swarm_obj = orchestrator.get_swarm(swarm)
    if not swarm_obj:
        console.print(f"[red]Swarm '{swarm}' not found.[/red]")
        return

    console.print(f"[dim]Sending directive to {swarm}...[/dim]")

    try:
        response = asyncio.run(orchestrator.send_directive(swarm, directive))
        console.print(Panel(
            Markdown(response),
            title=f"[bold]{swarm} Response[/bold]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


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
