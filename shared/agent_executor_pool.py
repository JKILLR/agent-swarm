"""Agent executor pool for concurrent agent execution.

This module provides the AgentExecutorPool class that manages concurrent
Claude CLI processes with proper isolation, resource limits, and event streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from .execution_context import AgentExecutionContext
from .workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


class AgentExecutorPool:
    """Pool for managing concurrent agent executions.

    Manages multiple Claude CLI processes with:
    - Concurrency limiting via semaphore
    - Process lifecycle tracking
    - Event streaming
    - Cancellation support
    - Event broadcasting to external listeners
    - COO WebSocket execution support

    Attributes:
        max_concurrent: Maximum number of concurrent agent processes
        workspace_manager: Manager for workspace isolation
        on_event: Optional callback for broadcasting events to external listeners
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        workspace_manager: WorkspaceManager | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        """Initialize the executor pool.

        Args:
            max_concurrent: Maximum concurrent agent executions
            workspace_manager: Optional workspace manager for path resolution
            on_event: Optional callback for broadcasting events (e.g., to WebSocket)
        """
        self.max_concurrent = max_concurrent
        self.workspace_manager = workspace_manager
        self.on_event = on_event  # External event broadcast callback
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._coo_execution_id: str | None = None  # Track COO execution

        logger.info(
            f"AgentExecutorPool initialized with max_concurrent={max_concurrent}"
        )

    def set_event_callback(self, callback: Callable[[dict], None] | None):
        """Set the event broadcast callback.

        Args:
            callback: Function to call with each event dict
        """
        self.on_event = callback

    def _broadcast_event(self, event: dict):
        """Broadcast an event to the external listener if configured."""
        event_type = event.get("type", "")
        if event_type in ("tool_start", "tool_complete"):
            logger.info(f"_broadcast_event called: {event_type} for {event.get('agent', '?')}, callback set: {self.on_event is not None}")
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Error broadcasting event: {e}")

    async def execute(
        self,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        disallowed_tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute an agent with the given context.

        Acquires a semaphore slot, runs the agent, and streams events.
        Ensures proper cleanup on completion or cancellation.

        Args:
            context: Execution context with all agent configuration
            prompt: The prompt to send to the agent
            system_prompt: Optional system prompt to append
            on_event: Optional callback for each event
            disallowed_tools: Optional list of tool names to disable (e.g., ["Write", "Edit"])

        Yields:
            Event dictionaries from the agent execution
        """
        execution_id = str(uuid.uuid4())

        logger.info(
            f"Starting execution {execution_id} for {context.full_name} "
            f"(max_turns={context.max_turns}, timeout={context.timeout})"
        )

        # Emit execution start event
        start_event = {
            "type": "agent_execution_start",
            "execution_id": execution_id,
            "agent": context.agent_name,
            "agent_type": context.agent_type,
            "swarm": context.swarm_name,
            "workspace": str(context.workspace),
        }
        if on_event:
            on_event(start_event)
        self._broadcast_event(start_event)  # Also broadcast to external listeners
        yield start_event

        async with self._semaphore:
            try:
                async for event in self._run_agent(
                    execution_id, context, prompt, system_prompt, disallowed_tools
                ):
                    if on_event:
                        on_event(event)
                    # Broadcast tool and progress events to external listeners
                    event_type = event.get("type", "")
                    if event_type in ("tool_start", "tool_complete", "thinking_start",
                                       "thinking_complete", "agent_execution_progress"):
                        self._broadcast_event(event)
                    yield event

                # Emit completion event
                complete_event = {
                    "type": "agent_execution_complete",
                    "execution_id": execution_id,
                    "agent": context.agent_name,
                    "success": True,
                    "result_summary": f"Agent {context.agent_name} completed successfully",
                }
                if on_event:
                    on_event(complete_event)
                self._broadcast_event(complete_event)
                yield complete_event

            except asyncio.CancelledError:
                logger.warning(f"Execution {execution_id} was cancelled")
                cancel_event = {
                    "type": "agent_execution_complete",
                    "execution_id": execution_id,
                    "agent": context.agent_name,
                    "success": False,
                    "result_summary": f"Agent {context.agent_name} was cancelled",
                }
                if on_event:
                    on_event(cancel_event)
                self._broadcast_event(cancel_event)
                yield cancel_event
                raise

            except Exception as e:
                logger.error(f"Execution {execution_id} failed: {e}")
                error_event = {
                    "type": "agent_execution_complete",
                    "execution_id": execution_id,
                    "agent": context.agent_name,
                    "success": False,
                    "result_summary": f"Agent {context.agent_name} failed: {e}",
                    "error": str(e),
                }
                if on_event:
                    on_event(error_event)
                self._broadcast_event(error_event)
                yield error_event

            finally:
                self._cleanup(execution_id)

    async def _run_agent(
        self,
        execution_id: str,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None,
        disallowed_tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a single agent execution.

        Args:
            execution_id: Unique ID for this execution
            context: Agent execution context
            prompt: The prompt to send
            system_prompt: Optional system prompt
            disallowed_tools: Optional list of tools to disable

        Yields:
            Event dictionaries parsed from the CLI output
        """
        # Build command
        cmd = self._build_command(context, prompt, system_prompt, disallowed_tools)

        # Build environment
        env = self._build_environment(context)

        logger.debug(
            f"Starting process for {context.full_name} in {context.workspace}"
        )

        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(context.workspace),
            env=env,
        )

        self._processes[execution_id] = process

        # Stream output with timeout
        try:
            async for event in self._parse_stream(process, context, execution_id):
                yield event

            await asyncio.wait_for(
                process.wait(),
                timeout=context.timeout
            )

            # Check for errors
            if process.returncode != 0:
                if process.stderr:
                    stderr = await process.stderr.read()
                    if stderr:
                        error_msg = stderr.decode().strip()
                        logger.error(
                            f"Agent {context.full_name} exited with code "
                            f"{process.returncode}: {error_msg[:200]}"
                        )
                        yield {
                            "type": "error",
                            "execution_id": execution_id,
                            "agent": context.agent_name,
                            "content": error_msg,
                        }

        except asyncio.TimeoutError:
            logger.error(
                f"Agent {context.full_name} timed out after {context.timeout}s"
            )
            process.kill()
            await process.wait()
            yield {
                "type": "error",
                "execution_id": execution_id,
                "agent": context.agent_name,
                "content": f"Execution timed out after {context.timeout} seconds",
            }

    def _build_command(
        self,
        context: AgentExecutionContext,
        prompt: str,
        system_prompt: str | None,
        disallowed_tools: list[str] | None = None,
    ) -> list[str]:
        """Build the Claude CLI command.

        Args:
            context: Agent execution context
            prompt: The user prompt
            system_prompt: Optional system prompt
            disallowed_tools: Optional list of tools to disable

        Returns:
            Command as list of strings
        """
        cmd = [
            "claude",
            "-p",  # Print mode (non-interactive)
            "--output-format", "stream-json",
            "--verbose",
            "--permission-mode", context.permission_mode,
            "--tools", "default",  # Give all tools - no restrictions
        ]

        # Add max turns limit
        cmd.extend(["--max-turns", str(context.max_turns)])

        # Add tool restrictions only if explicitly specified
        if disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])

        # Add system prompt if provided
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Add -- to stop flag parsing, then the prompt as final argument
        # This prevents variadic flags from consuming the prompt
        cmd.append("--")
        cmd.append(prompt)

        return cmd

    def _build_environment(
        self,
        context: AgentExecutionContext,
    ) -> dict[str, str]:
        """Build environment variables for the agent.

        Args:
            context: Agent execution context

        Returns:
            Environment dictionary for the subprocess
        """
        env = os.environ.copy()

        # Remove API key to force CLI auth (Max subscription)
        env.pop("ANTHROPIC_API_KEY", None)

        # Add git credentials if allowed
        if context.git_credentials:
            # CLAUDE_CODE_OAUTH_TOKEN enables git operations
            oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            if oauth_token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
                logger.debug(f"Git credentials enabled for {context.full_name}")

        # Add workspace info for agent context (useful for debugging/logging)
        env["AGENT_WORKSPACE"] = str(context.workspace)
        env["AGENT_NAME"] = context.agent_name
        env["AGENT_SWARM"] = context.swarm_name
        env["AGENT_TYPE"] = context.agent_type

        # Add job ID if present
        if context.job_id:
            env["AGENT_JOB_ID"] = context.job_id

        # Add parent agent if present (for tracing)
        if context.parent_agent:
            env["AGENT_PARENT"] = context.parent_agent

        return env

    async def _parse_stream(
        self,
        process: asyncio.subprocess.Process,
        context: AgentExecutionContext,
        execution_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse streaming JSON from Claude CLI.

        Args:
            process: The subprocess running Claude CLI
            context: Agent execution context
            execution_id: Unique execution ID

        Yields:
            Parsed event dictionaries
        """
        if not process.stdout:
            return

        current_block_type = None
        full_response = ""
        full_thinking = ""

        # Start draining stderr to prevent buffer deadlock
        async def drain_stderr():
            if process.stderr:
                try:
                    while True:
                        chunk = await process.stderr.read(4096)
                        if not chunk:
                            break
                        stderr_text = chunk.decode(errors="ignore").strip()
                        if stderr_text:
                            logger.debug(
                                f"Agent {context.full_name} stderr: {stderr_text[:200]}"
                            )
                except Exception as e:
                    logger.debug(f"Stderr drain error: {e}")

        stderr_task = asyncio.create_task(drain_stderr())

        try:
            buffer = b""
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(4096),
                        timeout=60.0  # Per-read timeout
                    )
                    if not chunk:
                        break

                    buffer += chunk

                    # Process complete lines
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line_str = line.decode().strip()
                        if not line_str:
                            continue

                        try:
                            event = json.loads(line_str)
                            event_type = event.get("type", "")

                            # Debug: log all event types from CLI
                            if event_type:
                                logger.info(f"CLI event from {context.agent_name}: type={event_type}")

                            # Add execution context to events
                            event["execution_id"] = execution_id
                            event["agent"] = context.agent_name

                            # Process and transform events
                            if event_type == "content_block_start":
                                content_block = event.get("content_block", {})
                                current_block_type = content_block.get("type", "text")
                                if current_block_type == "thinking":
                                    yield {
                                        "type": "thinking_start",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                    }
                                elif current_block_type == "tool_use":
                                    tool_name = content_block.get("name", "unknown")
                                    tool_input = content_block.get("input", {})
                                    # Generate a description for the tool activity panel
                                    description = get_tool_description(tool_name, tool_input)
                                    yield {
                                        "type": "tool_start",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "tool": tool_name,
                                        "input": tool_input,
                                        "description": description,
                                    }

                            elif event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                delta_type = delta.get("type", "")

                                if delta_type == "thinking_delta":
                                    text = delta.get("thinking", "")
                                    full_thinking += text
                                    yield {
                                        "type": "thinking_delta",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "delta": text,
                                    }
                                elif delta_type == "text_delta":
                                    text = delta.get("text", "")
                                    full_response += text
                                    yield {
                                        "type": "agent_delta",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "agent_type": context.agent_type,
                                        "delta": text,
                                    }

                            elif event_type == "content_block_stop":
                                if current_block_type == "thinking":
                                    yield {
                                        "type": "thinking_complete",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "thinking": full_thinking,
                                    }
                                elif current_block_type == "tool_use":
                                    yield {
                                        "type": "tool_complete",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "success": True,
                                    }
                                current_block_type = None

                            elif event_type == "result":
                                result_text = event.get("result", "")
                                if result_text:
                                    yield {
                                        "type": "content",
                                        "execution_id": execution_id,
                                        "agent": context.agent_name,
                                        "content": result_text,
                                    }

                            elif event_type == "assistant":
                                # Handle assistant message blocks
                                message = event.get("message", {})
                                content_blocks = message.get("content", [])
                                for block in content_blocks:
                                    if block.get("type") == "text":
                                        text = block.get("text", "")
                                        if text:
                                            yield {
                                                "type": "agent_delta",
                                                "execution_id": execution_id,
                                                "agent": context.agent_name,
                                                "agent_type": context.agent_type,
                                                "delta": text,
                                            }
                                    elif block.get("type") == "thinking":
                                        thinking = block.get("thinking", "")
                                        if thinking:
                                            yield {
                                                "type": "thinking_delta",
                                                "execution_id": execution_id,
                                                "agent": context.agent_name,
                                                "delta": thinking,
                                            }

                            else:
                                # Pass through other event types
                                yield event

                        except json.JSONDecodeError:
                            # Plain text output - emit as content
                            yield {
                                "type": "content",
                                "execution_id": execution_id,
                                "agent": context.agent_name,
                                "content": line_str + "\n",
                            }

                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    # Emit progress event
                    yield {
                        "type": "agent_execution_progress",
                        "execution_id": execution_id,
                        "agent": context.agent_name,
                        "progress": "still running",
                        "activity": "processing",
                    }

            # Process remaining buffer
            if buffer:
                for line in buffer.decode().split("\n"):
                    line_str = line.strip()
                    if line_str:
                        try:
                            event = json.loads(line_str)
                            event["execution_id"] = execution_id
                            event["agent"] = context.agent_name
                            yield event
                        except json.JSONDecodeError:
                            yield {
                                "type": "content",
                                "execution_id": execution_id,
                                "agent": context.agent_name,
                                "content": line_str,
                            }

        finally:
            # Ensure stderr task completes
            try:
                await asyncio.wait_for(stderr_task, timeout=5.0)
            except asyncio.TimeoutError:
                stderr_task.cancel()

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running agent execution.

        Args:
            execution_id: The execution ID to cancel

        Returns:
            True if cancelled, False if not found
        """
        if execution_id in self._processes:
            process = self._processes[execution_id]
            logger.info(f"Cancelling execution {execution_id}")
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Process for {execution_id} did not terminate cleanly"
                )
            return True

        if execution_id in self._running:
            task = self._running[execution_id]
            task.cancel()
            return True

        return False

    def _cleanup(self, execution_id: str) -> None:
        """Clean up after execution.

        Args:
            execution_id: The execution ID to clean up
        """
        self._processes.pop(execution_id, None)
        self._running.pop(execution_id, None)
        logger.debug(f"Cleaned up execution {execution_id}")

    @property
    def active_count(self) -> int:
        """Get the number of active executions."""
        return len(self._processes)

    @property
    def available_slots(self) -> int:
        """Get the number of available execution slots."""
        return self.max_concurrent - self.active_count


def get_tool_description(tool_name: str, tool_input: dict) -> str:
    """Generate human-readable description for a tool call.

    This is a shared utility function for generating consistent tool descriptions
    across the codebase (used by both main.py websocket handler and AgentExecutorPool).

    Args:
        tool_name: Name of the tool being used
        tool_input: Dictionary of tool input parameters

    Returns:
        Human-readable description of the tool action
    """
    descriptions = {
        "Read": lambda i: f"Reading {i.get('file_path', 'file')[:60]}",
        "Write": lambda i: f"Writing to {i.get('file_path', 'file')[:60]}",
        "Edit": lambda i: f"Editing {i.get('file_path', 'file')[:60]}",
        "Bash": lambda i: f"Running: {i.get('command', '')[:50]}{'...' if len(i.get('command', '')) > 50 else ''}",
        "Glob": lambda i: f"Searching for {i.get('pattern', 'files')}",
        "Grep": lambda i: f"Searching for '{i.get('pattern', '')[:40]}'",
        "Task": lambda i: f"Delegating to {i.get('subagent_type', i.get('agent', 'agent'))}: {i.get('description', i.get('prompt', ''))[:50]}...",
        "WebSearch": lambda i: f"Searching web: {i.get('query', '')[:50]}",
        "WebFetch": lambda i: f"Fetching {i.get('url', 'URL')[:50]}",
    }

    if tool_name in descriptions:
        try:
            return descriptions[tool_name](tool_input)
        except Exception:
            pass

    return f"Using {tool_name}"


# Module-level singleton with thread-safe initialization
import threading

_pool: AgentExecutorPool | None = None
_pool_lock = threading.Lock()


def get_executor_pool(
    max_concurrent: int = 5,
    workspace_manager: WorkspaceManager | None = None,
) -> AgentExecutorPool:
    """Get or create the global executor pool.

    Thread-safe singleton pattern with double-checked locking.

    Args:
        max_concurrent: Maximum concurrent executions (used on first call)
        workspace_manager: Optional workspace manager

    Returns:
        The executor pool singleton
    """
    global _pool

    if _pool is None:
        with _pool_lock:
            # Double-check after acquiring lock
            if _pool is None:
                _pool = AgentExecutorPool(
                    max_concurrent=max_concurrent,
                    workspace_manager=workspace_manager,
                )

    return _pool
