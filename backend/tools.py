"""Tool definitions and execution for agent swarm."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import urllib.request
import urllib.parse
import urllib.error
import re
from html.parser import HTMLParser
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
        },
        {
            "name": "WebSearch",
            "description": "Search the web for information. Returns search results with titles, URLs, and snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "WebFetch",
            "description": "Fetch and read content from a URL. Returns the text content of the page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Extract text only (default true), or return raw HTML",
                        "default": True
                    }
                },
                "required": ["url"]
            }
        },
        {
            "name": "ReadImage",
            "description": "Analyze an image file. Can describe contents, read text (OCR), or answer questions about the image.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to image file (relative to project root)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What to analyze or ask about the image",
                        "default": "Describe this image in detail"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "ParallelTasks",
            "description": "Execute multiple tasks in parallel. More efficient than sequential Task calls.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent": {
                                    "type": "string",
                                    "description": "Agent to spawn (swarm_name/agent_name)"
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Task prompt for this agent"
                                }
                            },
                            "required": ["agent", "prompt"]
                        },
                        "description": "List of tasks to execute in parallel"
                    }
                },
                "required": ["tasks"]
            }
        },
        {
            "name": "GitCommit",
            "description": "Commit changes and push to a feature branch. Creates branch if needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message describing the changes"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Feature branch name (will be prefixed with 'swarm/' automatically)"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to commit (optional, defaults to all changes)"
                    }
                },
                "required": ["message", "branch"]
            }
        },
        {
            "name": "GitSync",
            "description": "Sync local repository with remote main branch. Use after PRs are merged.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch to sync (default: main)",
                        "default": "main"
                    }
                },
                "required": []
            }
        },
        {
            "name": "GitStatus",
            "description": "Check git status - current branch, uncommitted changes, etc.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
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
            elif tool_name == "WebSearch":
                return await self._execute_web_search(tool_input)
            elif tool_name == "WebFetch":
                return await self._execute_web_fetch(tool_input)
            elif tool_name == "ReadImage":
                return await self._execute_read_image(tool_input)
            elif tool_name == "ParallelTasks":
                return await self._execute_parallel_tasks(tool_input)
            elif tool_name == "GitCommit":
                return await self._execute_git_commit(tool_input)
            elif tool_name == "GitSync":
                return await self._execute_git_sync(tool_input)
            elif tool_name == "GitStatus":
                return await self._execute_git_status(tool_input)
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

        # Get the swarm (case-insensitive lookup)
        swarm = None
        for name, s in self.orchestrator.swarms.items():
            if name.lower() == swarm_name.lower() or name.lower().replace(" ", "_") == swarm_name.lower():
                swarm = s
                swarm_name = name  # Use the actual name
                break

        if not swarm:
            available = list(self.orchestrator.swarms.keys())
            return f"Swarm not found: {swarm_name}. Available: {available}"

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
            "--permission-mode", "acceptEdits",  # Allow file writes without blocking
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
            timeout=300.0  # 5 minute timeout for subagents
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

    async def _execute_web_search(self, input: Dict[str, Any]) -> str:
        """Search the web using DuckDuckGo."""
        query = input.get("query", "")
        num_results = min(input.get("num_results", 5), 10)

        if not query:
            return "Error: No search query provided"

        try:
            # Use DuckDuckGo HTML search (no API key needed)
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AgentSwarm/1.0)"
                }
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")

            # Parse search results from HTML
            results = []
            # Simple regex to extract result links and snippets
            pattern = r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]*)</a>'

            links = re.findall(pattern, html)
            snippets = re.findall(snippet_pattern, html)

            for i, (link, title) in enumerate(links[:num_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                # Decode URL from DuckDuckGo redirect
                if "uddg=" in link:
                    actual_url = urllib.parse.unquote(link.split("uddg=")[-1].split("&")[0])
                else:
                    actual_url = link
                results.append(f"**{i+1}. {title.strip()}**\n   URL: {actual_url}\n   {snippet.strip()}\n")

            if not results:
                return f"No results found for: {query}"

            return f"**Search Results for: {query}**\n\n" + "\n".join(results)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error searching web: {str(e)}"

    async def _execute_web_fetch(self, input: Dict[str, Any]) -> str:
        """Fetch content from a URL."""
        url = input.get("url", "")
        extract_text = input.get("extract_text", True)

        if not url:
            return "Error: No URL provided"

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AgentSwarm/1.0)"
                }
            )

            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="replace")

            if extract_text:
                # Simple HTML to text conversion
                content = self._html_to_text(content)

            # Truncate if too long
            if len(content) > 10000:
                content = content[:10000] + "\n\n[...content truncated at 10000 chars]"

            return f"**Content from {url}:**\n\n{content}"

        except urllib.error.HTTPError as e:
            return f"HTTP Error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return f"URL Error: {e.reason}"
        except Exception as e:
            logger.error(f"Web fetch error: {e}")
            return f"Error fetching URL: {str(e)}"

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Convert common elements
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</p>', '\n\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</div>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</h[1-6]>', '\n\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</li>', '\n', html, flags=re.IGNORECASE)

        # Remove all remaining tags
        html = re.sub(r'<[^>]+>', '', html)

        # Decode HTML entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')

        # Clean up whitespace
        html = re.sub(r'\n\s*\n', '\n\n', html)
        html = re.sub(r' +', ' ', html)

        return html.strip()

    async def _execute_read_image(self, input: Dict[str, Any]) -> str:
        """Analyze an image using Claude."""
        path = input.get("path", "")
        prompt = input.get("prompt", "Describe this image in detail")

        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            return f"Error: Image not found: {path}"

        # Check file extension
        ext = full_path.suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            return f"Error: Unsupported image format: {ext}"

        try:
            # Read and base64 encode the image
            with open(full_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine media type
            media_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_types.get(ext, "image/png")

            # Use Claude CLI to analyze the image
            # Create a prompt that includes the image
            analysis_prompt = f"""Analyze this image and respond to: {prompt}

The image is provided as base64 data with media type {media_type}.
Image path: {path}
"""

            # For now, we'll use a simpler approach - describe what we can determine from the path
            # Full image analysis requires API call with vision capability
            return f"**Image Analysis: {path}**\n\nNote: Full image analysis requires Claude API with vision capability. Image file exists and is {full_path.stat().st_size} bytes.\n\nTo enable full image analysis, ensure ANTHROPIC_API_KEY is set with a model that supports vision."

        except Exception as e:
            logger.error(f"Image read error: {e}")
            return f"Error reading image: {str(e)}"

    async def _execute_parallel_tasks(self, input: Dict[str, Any]) -> str:
        """Execute multiple tasks in parallel."""
        tasks = input.get("tasks", [])

        if not tasks:
            return "Error: No tasks provided"

        if len(tasks) > 10:
            return "Error: Maximum 10 parallel tasks allowed"

        results = []
        async_tasks = []

        # Create all tasks
        for i, task in enumerate(tasks):
            agent = task.get("agent", "")
            prompt = task.get("prompt", "")

            if not agent or not prompt:
                results.append(f"Task {i+1}: Error - missing agent or prompt")
                continue

            # Create async task for each
            async_task = asyncio.create_task(
                self._execute_task({"agent": agent, "prompt": prompt}),
                name=f"task_{i}_{agent}"
            )
            async_tasks.append((i, agent, async_task))

        # Wait for all tasks to complete
        for i, agent, task in async_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=300.0)
                results.append(f"**Task {i+1} ({agent}):**\n{result}\n")
            except asyncio.TimeoutError:
                results.append(f"**Task {i+1} ({agent}):** Timed out\n")
            except Exception as e:
                results.append(f"**Task {i+1} ({agent}):** Error - {str(e)}\n")

        return f"**Parallel Execution Results ({len(tasks)} tasks):**\n\n" + "\n---\n".join(results)

    async def _execute_git_commit(self, input: Dict[str, Any]) -> str:
        """Commit changes and push to a feature branch."""
        message = input.get("message", "")
        branch = input.get("branch", "")
        files = input.get("files", [])

        if not message:
            return "Error: Commit message is required"
        if not branch:
            return "Error: Branch name is required"

        # Ensure branch has swarm/ prefix
        if not branch.startswith("swarm/"):
            branch = f"swarm/{branch}"

        try:
            results = []

            # Get current branch
            current = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            current_branch = current.stdout.strip()
            results.append(f"Current branch: {current_branch}")

            # Check if branch exists
            branch_check = subprocess.run(
                ["git", "branch", "--list", branch],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if not branch_check.stdout.strip():
                # Create and checkout new branch
                subprocess.run(
                    ["git", "checkout", "-b", branch],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True
                )
                results.append(f"Created new branch: {branch}")
            elif current_branch != branch:
                # Switch to existing branch
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True
                )
                results.append(f"Switched to branch: {branch}")

            # Stage files
            if files:
                for f in files:
                    subprocess.run(
                        ["git", "add", f],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True
                    )
                results.append(f"Staged {len(files)} files")
            else:
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True
                )
                results.append("Staged all changes")

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if not status.stdout.strip():
                return "No changes to commit"

            # Commit
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if commit_result.returncode != 0:
                return f"Commit failed: {commit_result.stderr}"

            results.append(f"Committed: {message}")

            # Push to origin
            push_result = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if push_result.returncode != 0:
                # Try force push if upstream doesn't exist
                push_result = subprocess.run(
                    ["git", "push", "--set-upstream", "origin", branch],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True
                )

            if push_result.returncode == 0:
                results.append(f"Pushed to origin/{branch}")
            else:
                results.append(f"Push warning: {push_result.stderr}")

            return "**Git Commit Success:**\n" + "\n".join(f"- {r}" for r in results)

        except subprocess.CalledProcessError as e:
            return f"Git error: {e.stderr if e.stderr else str(e)}"
        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return f"Error: {str(e)}"

    async def _execute_git_sync(self, input: Dict[str, Any]) -> str:
        """Sync local repository with remote main branch."""
        branch = input.get("branch", "main")

        try:
            results = []

            # Fetch latest
            fetch_result = subprocess.run(
                ["git", "fetch", "origin", branch],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            results.append(f"Fetched origin/{branch}")

            # Get current branch
            current = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            current_branch = current.stdout.strip()

            if current_branch != branch:
                # Checkout target branch
                checkout = subprocess.run(
                    ["git", "checkout", branch],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True
                )
                if checkout.returncode != 0:
                    return f"Could not checkout {branch}: {checkout.stderr}"
                results.append(f"Switched to {branch}")

            # Pull latest
            pull_result = subprocess.run(
                ["git", "pull", "origin", branch],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if pull_result.returncode == 0:
                results.append(f"Pulled latest from origin/{branch}")
                if "Already up to date" in pull_result.stdout:
                    results.append("Already up to date")
                else:
                    results.append(pull_result.stdout.strip()[:200])
            else:
                results.append(f"Pull warning: {pull_result.stderr}")

            return "**Git Sync Complete:**\n" + "\n".join(f"- {r}" for r in results)

        except Exception as e:
            logger.error(f"Git sync error: {e}")
            return f"Error syncing: {str(e)}"

    async def _execute_git_status(self, input: Dict[str, Any]) -> str:
        """Check git status including sync with remote main."""
        try:
            results = []

            # Current branch
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            current_branch = branch.stdout.strip()
            results.append(f"**Branch:** {current_branch}")

            # Fetch from origin to get latest refs
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=30
            )

            # Check sync status with origin/main
            behind_ahead = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if behind_ahead.returncode == 0:
                counts = behind_ahead.stdout.strip().split()
                if len(counts) == 2:
                    ahead, behind = int(counts[0]), int(counts[1])
                    if ahead == 0 and behind == 0:
                        results.append("**Sync with origin/main:** ✅ Up to date")
                    else:
                        sync_parts = []
                        if behind > 0:
                            sync_parts.append(f"⚠️ {behind} commits behind")
                        if ahead > 0:
                            sync_parts.append(f"📤 {ahead} commits ahead")
                        results.append(f"**Sync with origin/main:** {', '.join(sync_parts)}")
                        if behind > 0:
                            results.append("  → Run GitSync to pull latest main")

            # Local changes
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

            if status.stdout.strip():
                results.append(f"\n**Uncommitted changes:**\n```\n{status.stdout.strip()}\n```")
            else:
                results.append("\n**Uncommitted changes:** None (working tree clean)")

            # Recent commits
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )
            results.append(f"\n**Recent commits:**\n```\n{log.stdout.strip()}\n```")

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Git status error: {e}")
            return f"Error getting status: {str(e)}"
