"""Tool definitions and execution for agent swarm."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
import logging

logger = logging.getLogger(__name__)

# Project root for file operations
PROJECT_ROOT = Path(__file__).parent.parent


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions for Claude API."""
    return [
        {
            "name": "Task",
            "description": "Spawn a subagent to handle a task. Use this to delegate work to swarm agents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Agent to spawn, format: 'swarm_name/agent_name' (e.g., 'swarm_dev/implementer') or just 'agent_name' for operations"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task/prompt for the subagent"
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run in background (don't wait for result)",
                        "default": False
                    }
                },
                "required": ["agent", "prompt"]
            }
        },
        {
            "name": "Read",
            "description": "Read a file from the workspace",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (relative to project root)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "Write",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file (relative to project root)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "Bash",
            "description": "Execute a bash command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (relative to project root)"
                    }
                },
                "required": ["command"]
            }
        },
        {
            "name": "Glob",
            "description": "Find files matching a pattern",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base path to search from"
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "Grep",
            "description": "Search for text in files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to search in"
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py')"
                    }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "ListSwarms",
            "description": "List all available swarms and their agents",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "GetSwarmStatus",
            "description": "Get detailed status of a specific swarm",
            "input_schema": {
                "type": "object",
                "properties": {
                    "swarm": {
                        "type": "string",
                        "description": "Name of the swarm"
                    }
                },
                "required": ["swarm"]
            }
        }
    ]


class ToolExecutor:
    """Executes tools for agents."""

    def __init__(self, orchestrator, websocket=None, manager=None):
        self.orchestrator = orchestrator
        self.websocket = websocket
        self.manager = manager
        self.background_tasks: Dict[str, asyncio.Task] = {}

    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send event to websocket if available."""
        if self.websocket and self.manager:
            await self.manager.send_event(self.websocket, event_type, data)

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        try:
            if tool_name == "Task":
                return await self._execute_task(tool_input)
            elif tool_name == "Read":
                return await self._execute_read(tool_input)
            elif tool_name == "Write":
                return await self._execute_write(tool_input)
            elif tool_name == "Bash":
                return await self._execute_bash(tool_input)
            elif tool_name == "Glob":
                return await self._execute_glob(tool_input)
            elif tool_name == "Grep":
                return await self._execute_grep(tool_input)
            elif tool_name == "ListSwarms":
                return await self._execute_list_swarms(tool_input)
            elif tool_name == "GetSwarmStatus":
                return await self._execute_get_swarm_status(tool_input)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def _execute_task(self, input: Dict[str, Any]) -> str:
        """Spawn a subagent to handle a task."""
        agent_path = input.get("agent", "")
        prompt = input.get("prompt", "")
        background = input.get("background", False)

        # Parse agent path (swarm/agent or just agent)
        if "/" in agent_path:
            swarm_name, agent_name = agent_path.split("/", 1)
        else:
            # Default to operations swarm
            swarm_name = "operations"
            agent_name = agent_path

        # Get the swarm
        swarm = self.orchestrator.get_swarm(swarm_name)
        if not swarm:
            return f"Swarm not found: {swarm_name}"

        # Get the agent
        agent = swarm.get_agent(agent_name)
        if not agent:
            # List available agents
            available = list(swarm.agents.keys())
            return f"Agent '{agent_name}' not found in {swarm_name}. Available: {available}"

        # Notify that we're spawning an agent
        await self.send_event("agent_start", {
            "agent": f"{swarm_name}/{agent_name}",
            "agent_type": agent.role,
        })

        # Build context for the subagent
        agent_prompt = f"""You are {agent_name} in the {swarm_name} swarm.

Your role: {agent.role}
Workspace: {swarm.workspace}

## Task from orchestrator:
{prompt}

Please complete this task. You have access to tools: Read, Write, Bash, Glob, Grep.
"""

        if background:
            # Run in background - don't wait
            task_id = f"{swarm_name}-{agent_name}-{id(prompt)}"
            # For now, just acknowledge - full background execution would need more infrastructure
            return f"[Background task started: {task_id}] Agent {agent_name} is working on the task."

        # Execute the agent synchronously using Claude API
        try:
            result = await self._run_subagent(
                agent_name=f"{swarm_name}/{agent_name}",
                agent_role=agent.role,
                prompt=agent_prompt,
                workspace=swarm.workspace,
            )

            await self.send_event("agent_complete", {
                "agent": f"{swarm_name}/{agent_name}",
                "agent_type": agent.role,
                "content": result[:500] + "..." if len(result) > 500 else result,
            })

            return result
        except Exception as e:
            error_msg = f"Agent {agent_name} failed: {str(e)}"
            await self.send_event("agent_complete", {
                "agent": f"{swarm_name}/{agent_name}",
                "agent_type": agent.role,
                "content": error_msg,
            })
            return error_msg

    async def _run_subagent(
        self,
        agent_name: str,
        agent_role: str,
        prompt: str,
        workspace: Path,
    ) -> str:
        """Run a subagent using Claude CLI (Max subscription) or API fallback."""

        # Try Claude CLI first (uses Max subscription)
        try:
            result = await self._run_subagent_cli(prompt, workspace)
            if result:
                return result
        except Exception as e:
            logger.warning(f"CLI subagent failed: {e}")

        # Fall back to API if available
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return await self._run_subagent_api(prompt, workspace)

        return f"[Subagent execution failed - CLI not available and no API key set]"

    async def _run_subagent_cli(self, prompt: str, workspace: Path) -> str:
        """Run subagent via Claude CLI (uses Max subscription)."""
        import asyncio

        cmd = [
            "claude",
            "-p",  # Print mode
            "--output-format", "json",
            prompt,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace) if workspace else None,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=120.0  # 2 minute timeout for subagents
        )

        if process.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"CLI failed: {error}")

        # Parse JSON output
        output = stdout.decode()
        try:
            import json
            data = json.loads(output)
            # Extract text from response
            if isinstance(data, dict):
                if "result" in data:
                    return data["result"]
                if "content" in data:
                    return data["content"]
            return output
        except json.JSONDecodeError:
            return output

    async def _run_subagent_api(self, prompt: str, workspace: Path) -> str:
        """Run subagent via Anthropic API (fallback)."""
        try:
            import anthropic
        except ImportError:
            return "[Anthropic SDK not installed]"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "[ANTHROPIC_API_KEY not set]"

        client = anthropic.Anthropic(api_key=api_key)

        # Simpler tool set for subagents
        subagent_tools = [
            t for t in get_tool_definitions()
            if t["name"] in ["Read", "Write", "Bash", "Glob", "Grep"]
        ]

        messages = [{"role": "user", "content": prompt}]

        # Agentic loop for subagent (max 10 iterations)
        for i in range(10):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=subagent_tools,
                messages=messages,
            )

            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract text response
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text
                return text_content or "[Agent completed with no text output]"

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant message
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await self.execute(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Add tool results
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unknown stop reason
                return f"[Agent stopped unexpectedly: {response.stop_reason}]"

        return "[Agent reached maximum iterations]"

    async def _execute_read(self, input: Dict[str, Any]) -> str:
        """Read a file."""
        path = input.get("path", "")
        full_path = PROJECT_ROOT / path

        if not full_path.exists():
            return f"File not found: {path}"

        try:
            content = full_path.read_text()
            # Truncate if too long
            if len(content) > 10000:
                return content[:10000] + f"\n\n[...truncated, {len(content)} total chars]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    async def _execute_write(self, input: Dict[str, Any]) -> str:
        """Write to a file."""
        path = input.get("path", "")
        content = input.get("content", "")
        full_path = PROJECT_ROOT / path

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"Successfully wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def _execute_bash(self, input: Dict[str, Any]) -> str:
        """Execute a bash command."""
        command = input.get("command", "")
        cwd = input.get("cwd", "")

        work_dir = PROJECT_ROOT / cwd if cwd else PROJECT_ROOT

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"

            # Truncate if too long
            if len(output) > 5000:
                output = output[:5000] + "\n[...truncated]"

            return output or "[No output]"
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds"
        except Exception as e:
            return f"Error executing command: {e}"

    async def _execute_glob(self, input: Dict[str, Any]) -> str:
        """Find files matching a pattern."""
        pattern = input.get("pattern", "")
        base_path = input.get("path", "")

        search_path = PROJECT_ROOT / base_path if base_path else PROJECT_ROOT

        try:
            matches = list(search_path.glob(pattern))
            if not matches:
                return f"No files matching: {pattern}"

            # Return relative paths
            result = []
            for m in matches[:100]:  # Limit to 100
                try:
                    result.append(str(m.relative_to(PROJECT_ROOT)))
                except ValueError:
                    result.append(str(m))

            output = "\n".join(result)
            if len(matches) > 100:
                output += f"\n[...and {len(matches) - 100} more]"

            return output
        except Exception as e:
            return f"Error searching: {e}"

    async def _execute_grep(self, input: Dict[str, Any]) -> str:
        """Search for text in files."""
        pattern = input.get("pattern", "")
        path = input.get("path", ".")
        include = input.get("include", "")

        search_path = PROJECT_ROOT / path

        cmd = f"grep -r -n '{pattern}' {search_path}"
        if include:
            cmd = f"grep -r -n --include='{include}' '{pattern}' {search_path}"

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if not output:
                return f"No matches for: {pattern}"

            # Truncate if too long
            if len(output) > 5000:
                output = output[:5000] + "\n[...truncated]"

            return output
        except subprocess.TimeoutExpired:
            return "Search timed out"
        except Exception as e:
            return f"Error searching: {e}"

    async def _execute_list_swarms(self, input: Dict[str, Any]) -> str:
        """List all swarms and their agents."""
        result = []
        for name, swarm in self.orchestrator.swarms.items():
            status = swarm.get_status()
            agents = list(swarm.agents.keys())
            result.append(f"**{name}** ({status.get('status', 'unknown')})")
            result.append(f"  Description: {status.get('description', 'N/A')}")
            result.append(f"  Agents: {', '.join(agents)}")
            result.append("")

        return "\n".join(result) or "No swarms found"

    async def _execute_get_swarm_status(self, input: Dict[str, Any]) -> str:
        """Get detailed status of a swarm."""
        swarm_name = input.get("swarm", "")
        swarm = self.orchestrator.get_swarm(swarm_name)

        if not swarm:
            return f"Swarm not found: {swarm_name}"

        status = swarm.get_status()

        result = [
            f"# {status.get('name', swarm_name)}",
            f"**Status:** {status.get('status', 'unknown')}",
            f"**Description:** {status.get('description', 'N/A')}",
            f"**Version:** {status.get('version', 'N/A')}",
            "",
            "## Agents:",
        ]

        for agent_info in status.get("agents", []):
            result.append(f"- **{agent_info.get('name')}** ({agent_info.get('role', 'worker')})")

        result.append("")
        result.append("## Priorities:")
        for p in status.get("priorities", []):
            if isinstance(p, dict):
                result.append(f"- {p.get('task', str(p))}: {p.get('status', 'unknown')}")
            else:
                result.append(f"- {p}")

        return "\n".join(result)
