"""Agent executor using Claude CLI or Anthropic API.

This module provides real execution capability for agents, replacing the
mock claude_agent_sdk that doesn't exist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Check for Anthropic SDK
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


@dataclass
class ExecutionResult:
    """Result from agent execution."""

    content: str
    thinking: str = ""
    tool_uses: list[dict[str, Any]] = None
    success: bool = True
    error: str | None = None

    def __post_init__(self):
        if self.tool_uses is None:
            self.tool_uses = []


class AgentExecutor:
    """Executes agents using Claude CLI or Anthropic API."""

    def __init__(
        self,
        workspace: Path | None = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 300.0,
    ):
        """Initialize the executor.

        Args:
            workspace: Working directory for agent execution
            model: Model to use for API calls
            timeout: Timeout in seconds
        """
        self.workspace = workspace or Path.cwd()
        self.model = model
        self.timeout = timeout

        # Get auth token from environment
        self.oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute an agent with streaming output.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            tools: List of allowed tools

        Yields:
            Events with type and content
        """
        # Combine system prompt with prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        # Try Anthropic API first if available
        if self.api_key and ANTHROPIC_AVAILABLE:
            async for event in self._execute_anthropic(full_prompt, tools):
                yield event
        else:
            # Fall back to Claude CLI
            async for event in self._execute_cli(full_prompt, tools):
                yield event

    async def execute_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute an agent and return complete result.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            tools: List of allowed tools

        Returns:
            ExecutionResult with content and metadata
        """
        content_parts = []
        thinking_parts = []
        tool_uses = []
        error = None

        try:
            async for event in self.execute(prompt, system_prompt, tools):
                event_type = event.get("type", "")
                if event_type == "content":
                    content_parts.append(event.get("content", ""))
                elif event_type == "thinking":
                    thinking_parts.append(event.get("content", ""))
                elif event_type == "tool_use":
                    tool_uses.append(event)
                elif event_type == "error":
                    error = event.get("content", "Unknown error")
        except Exception as e:
            error = str(e)

        return ExecutionResult(
            content="".join(content_parts),
            thinking="".join(thinking_parts),
            tool_uses=tool_uses,
            success=error is None,
            error=error,
        )

    async def _execute_anthropic(
        self,
        prompt: str,
        tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute using Anthropic API with streaming."""
        if not anthropic:
            yield {"type": "error", "content": "Anthropic SDK not available"}
            return

        client = anthropic.Anthropic(api_key=self.api_key)

        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            block = event.content_block
                            if hasattr(block, "type") and block.type == "thinking":
                                yield {"type": "thinking_start"}

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, "type"):
                                if delta.type == "thinking_delta":
                                    yield {
                                        "type": "thinking",
                                        "content": delta.thinking,
                                    }
                                elif delta.type == "text_delta":
                                    yield {
                                        "type": "content",
                                        "content": delta.text,
                                    }

                        elif event.type == "content_block_stop":
                            yield {"type": "block_complete"}

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            yield {"type": "error", "content": str(e)}

    async def _execute_cli(
        self,
        prompt: str,
        tools: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute using Claude CLI with streaming JSON output."""
        # Build command
        cmd = [
            "claude",
            "-p",  # Print mode
            "--output-format",
            "stream-json",
            "--verbose",
            prompt,
        ]

        # Add permission mode if tools specified
        if tools:
            cmd.extend(["--permission-mode", "default"])

        # Build environment
        env = os.environ.copy()
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env=env,
            )

            # Read stdout line by line
            async for event in self._parse_cli_stream(process):
                yield event

            # Wait for process to complete
            await process.wait()

            # Check for errors
            if process.returncode != 0 and process.stderr:
                stderr = await process.stderr.read()
                if stderr:
                    error_msg = stderr.decode().strip()
                    logger.error(f"Claude CLI error: {error_msg}")
                    yield {"type": "error", "content": error_msg}

        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "content": "Claude CLI timed out. Check CLAUDE_CODE_OAUTH_TOKEN.",
            }
        except FileNotFoundError:
            yield {
                "type": "error",
                "content": "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-cli",
            }
        except Exception as e:
            logger.error(f"CLI execution error: {e}")
            yield {"type": "error", "content": str(e)}

    async def _parse_cli_stream(
        self,
        process: asyncio.subprocess.Process,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse streaming JSON from Claude CLI."""
        if not process.stdout:
            return

        current_block_type = None

        while True:
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                yield {"type": "error", "content": "Stream timeout"}
                break

            if not line:
                break

            line_str = line.decode().strip()
            if not line_str:
                continue

            try:
                event = json.loads(line_str)
                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    content_block = event.get("content_block", {})
                    current_block_type = content_block.get("type", "text")
                    if current_block_type == "thinking":
                        yield {"type": "thinking_start"}

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")

                    if delta_type == "thinking_delta":
                        yield {
                            "type": "thinking",
                            "content": delta.get("thinking", ""),
                        }
                    elif delta_type == "text_delta":
                        yield {
                            "type": "content",
                            "content": delta.get("text", ""),
                        }

                elif event_type == "content_block_stop":
                    yield {"type": "block_complete"}
                    current_block_type = None

                elif event_type == "tool_use":
                    yield {
                        "type": "tool_use",
                        "tool": event.get("name", ""),
                        "input": event.get("input", {}),
                    }

                elif event_type == "result":
                    result_text = event.get("result", "")
                    if result_text:
                        yield {"type": "content", "content": result_text}

            except json.JSONDecodeError:
                # Plain text output
                yield {"type": "content", "content": line_str + "\n"}


# Global executor instance
_executor: AgentExecutor | None = None


def get_executor(workspace: Path | None = None) -> AgentExecutor:
    """Get or create the global executor."""
    global _executor
    if _executor is None or (workspace and _executor.workspace != workspace):
        _executor = AgentExecutor(workspace=workspace)
    return _executor


async def execute_agent(
    prompt: str,
    system_prompt: str | None = None,
    tools: list[str] | None = None,
    workspace: Path | None = None,
) -> ExecutionResult:
    """Execute an agent and return the result.

    This is the main entry point for agent execution.

    Args:
        prompt: The prompt to send
        system_prompt: Optional system prompt
        tools: List of allowed tools
        workspace: Working directory

    Returns:
        ExecutionResult with content and metadata
    """
    executor = get_executor(workspace)
    return await executor.execute_sync(prompt, system_prompt, tools)


async def stream_agent(
    prompt: str,
    system_prompt: str | None = None,
    tools: list[str] | None = None,
    workspace: Path | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Execute an agent with streaming output.

    Args:
        prompt: The prompt to send
        system_prompt: Optional system prompt
        tools: List of allowed tools
        workspace: Working directory

    Yields:
        Events with type and content
    """
    executor = get_executor(workspace)
    async for event in executor.execute(prompt, system_prompt, tools):
        yield event
