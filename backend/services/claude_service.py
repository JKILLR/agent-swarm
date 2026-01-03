"""Claude CLI execution and streaming service.

This module provides functions to spawn and communicate with the Claude CLI
for AI-powered interactions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket
    from websocket.connection_manager import ConnectionManager

from .event_processor import CLIEventProcessor

logger = logging.getLogger(__name__)


async def stream_claude_response(
    prompt: str,
    swarm_name: str | None = None,
    workspace: Path | None = None,
    chat_id: str | None = None,
    system_prompt: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> asyncio.subprocess.Process:
    """Start a Claude CLI process and return it for streaming.

    Uses 'claude -p --output-format stream-json' which outputs JSON lines
    that we can parse and stream to the frontend.

    Args:
        prompt: The user message/request
        swarm_name: Optional swarm name for context
        workspace: Working directory for the CLI
        chat_id: Session ID for continuity
        system_prompt: Custom system prompt (COO role, context, etc.)
        disallowed_tools: List of tool names to disable (e.g., ["Write", "Edit"] for COO)

    Returns:
        The asyncio subprocess for streaming
    """
    # Build the command with prompt as argument (more reliable than stdin)
    cmd = [
        "claude",
        "-p",  # Print mode (non-interactive)
        "--output-format",
        "stream-json",
        "--verbose",  # Required for stream-json output
        "--permission-mode",
        "acceptEdits",  # Allow file writes without interactive approval
    ]

    # Add tool restrictions (e.g., COO cannot use Write/Edit)
    if disallowed_tools:
        cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])

    # Add custom system prompt for COO role (append to keep Claude's tool knowledge)
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    # NOTE: Session continuity disabled - was causing "session doesn't exist" errors
    # that confused the COO. Conversation history in prompt is sufficient context.

    # Add user prompt as final argument
    cmd.append(prompt)

    # Set working directory to workspace if specified
    cwd = str(workspace) if workspace else None

    # Build environment - remove API key so CLI uses Max subscription
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # Force CLI to use Max subscription

    logger.info(f"Starting Claude CLI in {cwd or 'current dir'}")

    # Start the process with explicit environment
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,  # Don't use stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    return process


async def parse_claude_stream(
    process: asyncio.subprocess.Process,
    websocket: "WebSocket",
    manager: "ConnectionManager",
    chat_id: str | None = None,
) -> dict:
    """Parse streaming JSON output from Claude CLI and send events to WebSocket.

    Captures session ID for continuity and returns dict with full response text and thinking.

    Args:
        process: The Claude CLI subprocess
        websocket: WebSocket connection to send events to
        manager: ConnectionManager for sending events
        chat_id: Optional chat session ID

    Returns:
        Dict with 'response' and 'thinking' keys containing accumulated text
    """
    # Create event processor for this stream
    processor = CLIEventProcessor(websocket, manager)

    # Session manager disabled - was causing confusion
    session_mgr = None

    if not process.stdout:
        return {"response": "", "thinking": ""}

    async def drain_stderr():
        """Drain stderr to prevent buffer deadlock."""
        if process.stderr:
            try:
                while True:
                    chunk = await process.stderr.read(65536)
                    if not chunk:
                        break
                    # Log stderr output for debugging
                    stderr_text = chunk.decode(errors='ignore').strip()
                    if stderr_text:
                        logger.debug(f"Claude CLI stderr: {stderr_text[:200]}")
            except Exception as e:
                logger.debug(f"Stderr drain error: {e}")

    # Start draining stderr concurrently to prevent deadlock
    stderr_task = asyncio.create_task(drain_stderr())

    # Read all output at once to avoid buffer issues, then parse line by line
    buffer = b""
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(process.stdout.read(65536), timeout=1.0)
                if not chunk:
                    break
                buffer += chunk

                # Process complete lines from buffer
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                        # Process event using the CLIEventProcessor
                        await processor.process(event, session_mgr, chat_id)
                    except json.JSONDecodeError:
                        continue
            except asyncio.TimeoutError:
                # Check if process is still running
                if process.returncode is not None:
                    break
                continue
            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                break

        # Process any remaining data in buffer
        if buffer:
            for line in buffer.decode().split("\n"):
                line_str = line.strip()
                if line_str:
                    try:
                        event = json.loads(line_str)
                        await processor.process(event, session_mgr, chat_id)
                    except json.JSONDecodeError:
                        pass
    finally:
        # Ensure stderr task completes
        try:
            await asyncio.wait_for(stderr_task, timeout=5.0)
        except asyncio.TimeoutError:
            stderr_task.cancel()

    # Wait for process to complete
    await process.wait()

    return processor.get_result()


def get_file_info(tool_name: str, tool_input: dict) -> tuple[str | None, str | None]:
    """Extract file path and operation type from tool input.

    Args:
        tool_name: Name of the tool being used
        tool_input: Tool input parameters

    Returns:
        Tuple of (file_path, operation_type) or (None, None)
    """
    file_path = tool_input.get('file_path')

    if tool_name == 'Read':
        return file_path, 'read'
    elif tool_name == 'Write':
        return file_path, 'write'
    elif tool_name == 'Edit':
        return file_path, 'edit'
    return None, None
