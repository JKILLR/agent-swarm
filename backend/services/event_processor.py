"""Process Claude CLI streaming events.

This module contains the CLIEventProcessor class which handles streaming
events from Claude CLI and forwards them to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket
    from websocket.connection_manager import ConnectionManager

from shared.agent_executor_pool import get_tool_description

logger = logging.getLogger(__name__)


class CLIEventProcessor:
    """Process streaming events from Claude CLI.

    This class manages the state for processing CLI events including:
    - Agent tracking stack (hierarchy of active agents)
    - Tool tracking for proper completion events
    - Response and thinking text accumulation
    """

    def __init__(self, websocket: "WebSocket", manager: "ConnectionManager"):
        """Initialize the event processor.

        Args:
            websocket: The WebSocket connection to send events to
            manager: The ConnectionManager for sending events
        """
        self.websocket = websocket
        self.manager = manager
        self.context: dict[str, Any] = {
            "full_response": "",
            "full_thinking": "",
            "current_block_type": None,
            "current_tool": None,
            "current_tool_use_id": "",
            "current_tool_input_json": "",
            "session_id": None,
            "agent_stack": ["COO"],  # COO is always the base
            "pending_tasks": {},
            "subagent_tools": {},
            "agent_spawn_sent": False,
        }

    def get_current_agent(self) -> str:
        """Get the currently active agent (top of stack)."""
        return self.context["agent_stack"][-1] if self.context["agent_stack"] else "COO"

    def get_result(self) -> dict[str, str]:
        """Get the accumulated response and thinking."""
        return {
            "response": self.context["full_response"],
            "thinking": self.context["full_thinking"],
        }

    async def process(self, event: dict, session_mgr=None, chat_id: str = None):
        """Process a single CLI event and forward to WebSocket.

        Agent Tracking Logic:
        - context["agent_stack"] tracks the hierarchy of active agents
        - When Task tool starts, we push the sub-agent name onto the stack
        - When Task tool completes, we pop it off
        - All tool events are attributed to the current top-of-stack agent

        Args:
            event: The CLI event to process
            session_mgr: Optional session manager for session continuity
            chat_id: Optional chat session ID
        """
        event_type = event.get("type", "")

        try:
            # Dispatch to appropriate handler
            handlers = {
                "assistant": self._handle_assistant,
                "content_block_start": self._handle_block_start,
                "content_block_delta": self._handle_block_delta,
                "content_block_stop": self._handle_block_stop,
                "result": self._handle_result,
                "tool_result": self._handle_tool_result,
                "user": self._handle_user,
            }

            # Handle session ID capture first
            if event_type in ("init", "system", "session_start") and session_mgr and chat_id:
                session_id = event.get("session_id") or event.get("sessionId")
                if session_id:
                    self.context["session_id"] = session_id
                    asyncio.create_task(session_mgr.register_session(chat_id, session_id))

            # Handle subagent assistant messages separately
            if event_type == "assistant" and event.get("parent_tool_use_id"):
                await self._handle_subagent_assistant(event)
                return

            handler = handlers.get(event_type)
            if handler:
                await handler(event, session_mgr, chat_id)

        except Exception as e:
            logger.error(f"Error processing CLI event: {e}")

    async def _handle_assistant(self, event: dict, session_mgr, chat_id):
        """Handle assistant message events.

        This processes the assistant's response content blocks including
        thinking, text, and tool_use blocks.
        """
        # Skip if this is a subagent message (has parent_tool_use_id)
        if event.get("parent_tool_use_id"):
            return

        message = event.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "thinking":
                text = block.get("thinking", "")
                if text:
                    self.context["full_thinking"] = self.context.get("full_thinking", "") + text

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    await self.manager.send_event(
                        self.websocket,
                        "agent_delta",
                        {
                            "agent": "Supreme Orchestrator",
                            "agent_type": "orchestrator",
                            "delta": text,
                        },
                    )
                    self.context["full_response"] = self.context.get("full_response", "") + text

            elif block_type == "tool_use":
                await self._handle_tool_use_block(block)

    async def _handle_tool_use_block(self, block: dict):
        """Handle a tool_use block from assistant message.

        This is a fallback handler for tool_use blocks that weren't
        captured during streaming events.
        """
        tool_name = block.get("name", "unknown")
        tool_input = block.get("input", {})
        tool_use_id = block.get("id", "")

        current_agent = self.get_current_agent()

        # Handle Task agent spawning if not already done via streaming
        if tool_name == "Task" and tool_use_id not in self.context.get("pending_tasks", {}):
            subagent = tool_input.get("subagent_type") or tool_input.get("agent", "")
            description = tool_input.get("description", tool_input.get("prompt", ""))[:100]

            if subagent:
                # Push to agent stack
                self.context["agent_stack"].append(subagent)
                # Track for cleanup
                self.context["pending_tasks"][tool_use_id] = subagent
                # Send agent_spawn event
                await self.manager.send_event(
                    self.websocket,
                    "agent_spawn",
                    {
                        "agent": subagent,
                        "description": description,
                        "parentAgent": current_agent,
                    },
                )
                current_agent = subagent  # Use subagent for this tool

        # Only send tool events if we didn't already send via streaming
        block_id = f"sent_tool_{tool_name}_{id(block)}"
        if not self.context.get(block_id):
            await self.manager.send_event(
                self.websocket,
                "tool_start",
                {
                    "tool": tool_name,
                    "description": get_tool_description(tool_name, tool_input),
                    "input": tool_input,
                    "agentName": current_agent,
                },
            )
            await self.manager.send_event(
                self.websocket,
                "tool_complete",
                {
                    "tool": tool_name,
                    "success": True,
                    "summary": f"Completed: {tool_name}",
                    "agentName": current_agent,
                },
            )
            self.context[block_id] = True

    async def _handle_block_start(self, event: dict, session_mgr, chat_id):
        """Handle content_block_start events."""
        content_block = event.get("content_block", {})
        block_type = content_block.get("type", "text")
        self.context["current_block_type"] = block_type

        if block_type == "thinking":
            await self.manager.send_event(
                self.websocket,
                "thinking_start",
                {"agent": "Supreme Orchestrator"},
            )
        elif block_type == "tool_use":
            tool_name = content_block.get("name", "unknown")
            tool_use_id = content_block.get("id", "")
            self.context["current_tool"] = tool_name
            self.context["current_tool_use_id"] = tool_use_id
            # Reset streamed input accumulator
            self.context["current_tool_input_json"] = ""
            self.context["agent_spawn_sent"] = False

            current_agent = self.get_current_agent()

            # Mark as sent to avoid duplicate from fallback
            self.context[f"sent_tool_{tool_name}"] = True

            # Send initial tool_start event (description updated when input available)
            await self.manager.send_event(
                self.websocket,
                "tool_start",
                {
                    "tool": tool_name,
                    "description": f"Starting {tool_name}...",
                    "input": {},
                    "agentName": current_agent,
                },
            )

    async def _handle_block_delta(self, event: dict, session_mgr, chat_id):
        """Handle content_block_delta events."""
        delta = event.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "thinking_delta":
            text = delta.get("thinking", "")
            self.context["full_thinking"] = self.context.get("full_thinking", "") + text
            await self.manager.send_event(
                self.websocket,
                "thinking_delta",
                {
                    "agent": "Supreme Orchestrator",
                    "delta": text,
                },
            )

        elif delta_type == "text_delta":
            text = delta.get("text", "")
            self.context["full_response"] = self.context.get("full_response", "") + text
            await self.manager.send_event(
                self.websocket,
                "agent_delta",
                {
                    "agent": "Supreme Orchestrator",
                    "agent_type": "orchestrator",
                    "delta": text,
                },
            )

        elif delta_type == "input_json_delta":
            await self._handle_input_json_delta(delta)

    async def _handle_input_json_delta(self, delta: dict):
        """Handle input_json_delta events for tool input streaming."""
        partial_json = delta.get("partial_json", "")
        self.context["current_tool_input_json"] = self.context.get("current_tool_input_json", "") + partial_json

        # Try to parse and detect agent spawning for Task tool
        if self.context.get("current_tool") == "Task" and not self.context.get("agent_spawn_sent"):
            try:
                partial_input = json.loads(self.context["current_tool_input_json"])
                # Check both subagent_type (standard) and agent (legacy) fields
                agent_name = partial_input.get("subagent_type") or partial_input.get("agent", "")

                if agent_name:
                    desc = partial_input.get("description", "") or partial_input.get("prompt", "")[:100]
                    parent_agent = self.get_current_agent()

                    # Push to agent stack for proper tool attribution
                    self.context["agent_stack"].append(agent_name)

                    # Track in pending_tasks for cleanup on completion
                    tool_use_id = self.context.get("current_tool_use_id", "")
                    self.context["pending_tasks"][tool_use_id] = agent_name

                    # Send agent_spawn event
                    await self.manager.send_event(
                        self.websocket,
                        "agent_spawn",
                        {
                            "agent": agent_name,
                            "description": desc,
                            "parentAgent": parent_agent,
                        },
                    )
                    self.context["agent_spawn_sent"] = True
            except json.JSONDecodeError:
                pass  # JSON not complete yet

    async def _handle_block_stop(self, event: dict, session_mgr, chat_id):
        """Handle content_block_stop events."""
        block_type = self.context.get("current_block_type")

        if block_type == "thinking":
            await self.manager.send_event(
                self.websocket,
                "thinking_complete",
                {
                    "agent": "Supreme Orchestrator",
                    "thinking": self.context.get("full_thinking", ""),
                },
            )

        elif block_type == "tool_use":
            tool_name = self.context.get("current_tool", "unknown")
            tool_use_id = self.context.get("current_tool_use_id", "")
            current_agent = self.get_current_agent()

            await self.manager.send_event(
                self.websocket,
                "tool_complete",
                {
                    "tool": tool_name,
                    "success": True,
                    "summary": f"Completed: {tool_name}",
                    "agentName": current_agent,
                },
            )

            # If this was a Task tool, pop the agent from stack
            if tool_name == "Task" and tool_use_id in self.context.get("pending_tasks", {}):
                completed_agent = self.context["pending_tasks"].pop(tool_use_id)
                if self.context["agent_stack"] and self.context["agent_stack"][-1] == completed_agent:
                    self.context["agent_stack"].pop()
                # Send agent completion event
                await self.manager.send_event(
                    self.websocket,
                    "agent_complete_subagent",
                    {
                        "agent": completed_agent,
                        "success": True,
                    },
                )

            # Reset tool context
            self.context["current_tool"] = None
            self.context["current_tool_use_id"] = ""
            self.context["current_tool_input_json"] = ""
            self.context["agent_spawn_sent"] = False

    async def _handle_result(self, event: dict, session_mgr, chat_id):
        """Handle final result events from CLI."""
        text = event.get("result", "")
        if text:
            # Always send the final result - this is the complete response
            self.context["full_response"] = text
            await self.manager.send_event(
                self.websocket,
                "agent_delta",
                {
                    "agent": "Supreme Orchestrator",
                    "agent_type": "orchestrator",
                    "delta": text,
                },
            )

    async def _handle_tool_result(self, event: dict, session_mgr, chat_id):
        """Handle tool_result events (explicit tool completion with error info)."""
        tool_name = self.context.get("current_tool", "unknown")
        success = event.get("is_error", False) is False
        await self.manager.send_event(
            self.websocket,
            "tool_complete",
            {
                "tool": tool_name,
                "success": success,
                "summary": f"{'Completed' if success else 'Failed'}: {tool_name}",
            },
        )
        self.context["current_tool"] = None

    async def _handle_user(self, event: dict, session_mgr, chat_id):
        """Handle user message events containing tool results.

        When parent_tool_use_id is set, this is a subagent's tool result.
        """
        parent_tool_id = event.get("parent_tool_use_id")
        message = event.get("message", {})
        content_list = message.get("content", [])

        if parent_tool_id and parent_tool_id in self.context.get("pending_tasks", {}):
            # This is a subagent's tool result
            subagent_name = self.context["pending_tasks"][parent_tool_id]

            for content_item in content_list:
                if content_item.get("type") == "tool_result":
                    tool_use_id = content_item.get("tool_use_id", "")
                    result_content = content_item.get("content", "")
                    is_error = content_item.get("is_error", False)

                    # Look up what tool this was from our tracking
                    tool_info = self.context.get("subagent_tools", {}).get(tool_use_id, {})
                    tool_name = tool_info.get("name", "Tool")

                    # Send tool_complete for subagent
                    await self.manager.send_event(
                        self.websocket,
                        "tool_complete",
                        {
                            "tool": tool_name,
                            "success": not is_error,
                            "summary": result_content[:100] if isinstance(result_content, str) else "Completed",
                            "agentName": subagent_name,
                        },
                    )

    async def _handle_subagent_assistant(self, event: dict):
        """Handle subagent assistant messages (tool calls from subagents)."""
        parent_tool_id = event.get("parent_tool_use_id")
        if parent_tool_id not in self.context.get("pending_tasks", {}):
            return

        subagent_name = self.context["pending_tasks"][parent_tool_id]
        message = event.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_input = block.get("input", {})
                tool_use_id = block.get("id", "")

                # Track this tool for later completion
                if "subagent_tools" not in self.context:
                    self.context["subagent_tools"] = {}
                self.context["subagent_tools"][tool_use_id] = {
                    "name": tool_name,
                    "agent": subagent_name,
                }

                # Emit tool_start for the subagent
                await self.manager.send_event(
                    self.websocket,
                    "tool_start",
                    {
                        "tool": tool_name,
                        "description": get_tool_description(tool_name, tool_input),
                        "input": tool_input,
                        "agentName": subagent_name,
                    },
                )

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    # Subagent is producing text output
                    await self.manager.send_event(
                        self.websocket,
                        "agent_delta",
                        {
                            "agent": subagent_name,
                            "agent_type": "subagent",
                            "delta": text,
                        },
                    )
