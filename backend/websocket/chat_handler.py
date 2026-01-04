"""Main chat WebSocket endpoint.

This module handles the primary chat WebSocket that connects
the frontend to the Supreme Orchestrator (COO).

All COO execution now routes through AgentExecutorPool for:
- Proper workspace isolation
- Consistent event streaming
- Resource management via semaphore
- Unified execution tracking
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from supreme.orchestrator import SupremeOrchestrator

from .connection_manager import manager
from ..services.chat_history import get_chat_history
from ..services.orchestrator_service import get_orchestrator
from shared.agent_executor_pool import get_executor_pool
from shared.execution_context import AgentExecutionContext

# NOTE: stream_claude_response and parse_claude_stream are no longer used
# All COO execution now goes through execute_coo_via_pool -> AgentExecutorPool

logger = logging.getLogger(__name__)

# Request correlation ID context variable
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def build_coo_system_prompt(orchestrator: "SupremeOrchestrator", project_root: Path) -> str:
    """Build the system prompt for the Supreme Orchestrator.

    Args:
        orchestrator: The SupremeOrchestrator instance
        project_root: Path to the project root

    Returns:
        The formatted system prompt string
    """
    all_swarms = []
    for name, s in orchestrator.swarms.items():
        agents_list = list(s.agents.keys())
        all_swarms.append(f"  - {name}: {', '.join(agents_list)}")

    all_swarms_str = "\n".join(all_swarms) if all_swarms else "  No swarms defined"

    return f"""You are the Supreme Orchestrator (COO) - a fully autonomous AI orchestrator.

## YOUR CAPABILITIES

You have FULL access to all tools including Write, Edit, Read, Bash, Glob, Grep, Task, and WebSearch.

You can either do work directly OR delegate to specialized agents. Use your judgment.

## PRIMARY DELEGATION: REST API (For Complex Tasks)

Use the REST API for complex implementation work that needs custom agent behavior:

```bash
curl -X POST http://localhost:8000/api/agents/execute \\
  -H "Content-Type: application/json" \\
  -d '{{"swarm": "swarm_dev", "agent": "implementer", "prompt": "Read workspace/STATE.md. Then implement X. Update STATE.md when done."}}'
```

### Available Swarm Agents (via REST API)

**swarm_dev** (for agent-swarm development):
- **implementer** - Write code, create/modify files
- **architect** - Design solutions, create plans
- **critic** - Review code for bugs/issues
- **reviewer** - Code review and quality checks

**operations** (for cross-swarm coordination):
- **ops_coordinator** - Multi-swarm coordination, status reports
- **qa_agent** - Quality audits, standards enforcement

## SECONDARY: Task Tool (Quick Tasks)

The Task tool works for quick operations:
- Web searches
- File reads and exploration
- Quick questions

```
Task(subagent_type="researcher", prompt="Search for X and summarize findings")
```

## Your Tools

You have FULL access to:
- **Read** - Read any file to understand context
- **Write** - Create new files
- **Edit** - Modify existing files
- **Glob/Grep** - Search files and code
- **Bash** - Run commands (git, tests, curl for REST API)
- **Task** - Delegate to built-in agents (researcher, architect, implementer, critic, tester)
- **WebSearch** - Search the web for information

## Standard Delegation Pipeline (Optional)

1. **swarm_dev/architect** → Design the solution
2. **swarm_dev/implementer** → Write the code
3. **swarm_dev/critic** → Review for bugs/issues
4. **swarm_dev/reviewer** → Final code review

## Swarm Workspaces
{all_swarms_str}
Files at: swarms/<swarm_name>/workspace/

## STATE.md - Shared Memory
- Read it before acting to understand current state
- Tell delegated agents to read and update it
- Update it after completing significant work

## Operations Reference
- Protocols: `swarms/operations/protocols/coordination_model.md`
- Quick reference: `swarms/operations/protocols/coo_quick_reference.md`

## Project Root: {project_root}

## Your Approach
1. Understand what the user wants
2. For simple tasks: Do them directly using your tools
3. For complex tasks: Delegate to specialized agents
4. Maintain natural conversation flow - remember context from earlier messages
5. Be direct and helpful

**You have full autonomy. Use your judgment on when to act directly vs delegate.**"""


async def execute_coo_via_pool(
    websocket: WebSocket,
    project_root: Path,
    prompt: str,
    system_prompt: str,
) -> dict[str, str]:
    """Execute COO request through AgentExecutorPool.

    Routes COO execution through the unified executor pool for:
    - Proper workspace isolation
    - Consistent event streaming
    - Resource management
    - Unified execution tracking

    Args:
        websocket: WebSocket connection for streaming events
        project_root: Path to project root (COO workspace)
        prompt: User prompt to process
        system_prompt: COO system prompt

    Returns:
        Dict with 'response' and 'thinking' keys
    """
    # Create COO execution context - COO has FULL tool access
    context = AgentExecutionContext(
        agent_name="coo",
        agent_type="orchestrator",
        swarm_name="supreme",  # COO is in the "supreme" namespace
        workspace=project_root,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
        git_credentials=True,  # COO needs git for reading repos
        web_access=True,
        max_turns=100,  # COO may need many turns for complex orchestration
        timeout=3600.0,  # 1 hour timeout
    )

    pool = get_executor_pool()

    full_response = ""
    full_thinking = ""
    current_block_type = None
    agent_stack = ["Supreme Orchestrator"]
    pending_tasks: dict[str, str] = {}

    try:
        async for event in pool.execute(
            context=context,
            prompt=prompt,
            system_prompt=system_prompt,
            disallowed_tools=None,  # COO has FULL tool access - no restrictions
        ):
            event_type = event.get("type", "")

            # Map pool events to WebSocket events
            if event_type == "agent_execution_start":
                # Already sent agent_start, skip this
                pass

            elif event_type == "thinking_start":
                await manager.send_event(
                    websocket,
                    "thinking_start",
                    {"agent": "Supreme Orchestrator"},
                )

            elif event_type == "thinking_delta":
                delta = event.get("delta", "")
                full_thinking += delta
                await manager.send_event(
                    websocket,
                    "thinking_delta",
                    {"agent": "Supreme Orchestrator", "delta": delta},
                )

            elif event_type == "thinking_complete":
                await manager.send_event(
                    websocket,
                    "thinking_complete",
                    {"agent": "Supreme Orchestrator", "thinking": full_thinking},
                )

            elif event_type == "agent_delta":
                delta = event.get("delta", "")
                full_response += delta
                await manager.send_event(
                    websocket,
                    "agent_delta",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "delta": delta,
                    },
                )

            elif event_type == "tool_start":
                tool_name = event.get("tool", "unknown")
                tool_input = event.get("input", {})
                description = event.get("description", f"Using {tool_name}")
                current_agent = agent_stack[-1] if agent_stack else "Supreme Orchestrator"

                # Handle Task tool spawning
                if tool_name == "Task":
                    subagent = tool_input.get("subagent_type") or tool_input.get("agent", "")
                    if subagent:
                        agent_stack.append(subagent)
                        task_desc = tool_input.get("description", tool_input.get("prompt", ""))[:100]
                        await manager.send_event(
                            websocket,
                            "agent_spawn",
                            {
                                "agent": subagent,
                                "description": task_desc,
                                "parentAgent": current_agent,
                            },
                        )
                        current_agent = subagent

                await manager.send_event(
                    websocket,
                    "tool_start",
                    {
                        "tool": tool_name,
                        "description": description,
                        "input": tool_input,
                        "agentName": current_agent,
                    },
                )

            elif event_type == "tool_complete":
                tool_name = event.get("tool", "unknown")
                success = event.get("success", True)
                current_agent = agent_stack[-1] if agent_stack else "Supreme Orchestrator"

                await manager.send_event(
                    websocket,
                    "tool_complete",
                    {
                        "tool": tool_name,
                        "success": success,
                        "summary": f"{'Completed' if success else 'Failed'}: {tool_name}",
                        "agentName": current_agent,
                    },
                )

                # Pop agent from stack if this was a Task tool completion
                if tool_name == "Task" and len(agent_stack) > 1:
                    completed_agent = agent_stack.pop()
                    await manager.send_event(
                        websocket,
                        "agent_complete_subagent",
                        {"agent": completed_agent, "success": success},
                    )

            elif event_type == "content":
                # Raw content - may be final response
                content = event.get("content", "")
                if content and not full_response:
                    full_response = content

            elif event_type == "agent_execution_complete":
                # Execution finished
                pass

            elif event_type == "error":
                error_content = event.get("content", "Unknown error")
                raise RuntimeError(f"Execution error: {error_content}")

    except asyncio.CancelledError:
        logger.warning("COO execution was cancelled")
        raise

    return {"response": full_response, "thinking": full_thinking}


async def websocket_chat(websocket: WebSocket, project_root: Path):
    """WebSocket endpoint for streaming chat.

    This is the main chat endpoint that connects the frontend to the
    Supreme Orchestrator (COO). It handles:
    - Message reception and acknowledgment
    - Image attachment processing
    - Conversation history management
    - Claude CLI execution and streaming
    - Session summary persistence

    Args:
        websocket: The WebSocket connection
        project_root: Path to the project root
    """
    await manager.connect(websocket)
    orch = get_orchestrator(project_root)
    history = get_chat_history(project_root)

    # Generate a connection-level correlation ID for the WebSocket session
    ws_correlation_id = str(uuid.uuid4())[:8]
    request_id_var.set(ws_correlation_id)
    logger.info("WebSocket chat session started")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            _swarm_name = data.get("swarm")  # Reserved for future swarm-specific routing
            session_id = data.get("session_id")
            attachments = data.get("attachments", [])

            # Generate a new correlation ID for each message in the WebSocket session
            msg_correlation_id = str(uuid.uuid4())[:8]
            request_id_var.set(msg_correlation_id)
            logger.info(f"Processing chat message: {message[:100]}...")

            if not message:
                continue

            # Process image attachments - save to temp files for Claude CLI
            image_paths = []
            if attachments:
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "agent_swarm_images"
                temp_dir.mkdir(exist_ok=True)

                for att in attachments:
                    if att.get("type") == "image" and att.get("content"):
                        # Save base64 image to temp file
                        img_data = base64.b64decode(att["content"])
                        ext = ".png"
                        if att.get("mimeType"):
                            if "jpeg" in att["mimeType"] or "jpg" in att["mimeType"]:
                                ext = ".jpg"
                            elif "gif" in att["mimeType"]:
                                ext = ".gif"
                            elif "webp" in att["mimeType"]:
                                ext = ".webp"
                        img_path = temp_dir / f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{att.get('name', 'image')}{ext}"
                        img_path.write_bytes(img_data)
                        image_paths.append(str(img_path))
                        logger.info(f"Saved image attachment to: {img_path}")

                # Add image paths to the message for Claude to read
                if image_paths:
                    message += f"\n\n[Images saved for analysis: {', '.join(image_paths)}]\nUse the Read tool to view these images."

            # Send acknowledgment
            await manager.send_event(
                websocket,
                "chat_start",
                {
                    "message": message,
                },
            )

            # Send thinking indicator
            await manager.send_event(
                websocket,
                "agent_start",
                {
                    "agent": "Supreme Orchestrator",
                    "agent_type": "orchestrator",
                },
            )

            try:
                # Import memory manager
                from memory import get_memory_manager
                memory = get_memory_manager()

                # Build conversation history - use last 10 messages for good context
                conversation_history = ""
                if session_id:
                    session = history.get_session(session_id)
                    if session and session.get("messages"):
                        messages = session["messages"]
                        # Use last 10 messages to maintain conversation flow
                        recent_messages = messages[-10:] if len(messages) > 10 else messages

                        history_lines = []
                        for msg in recent_messages:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            content = msg["content"]
                            # Truncate very long messages
                            if len(content) > 1000:
                                content = content[:1000] + "..."
                            history_lines.append(f"{role}: {content}")

                        if history_lines:
                            conversation_history = (
                                "\n\n## Recent Context\n" + "\n\n".join(history_lines) + "\n\n---\n"
                            )

                # Build system prompt for the COO
                system_prompt = build_coo_system_prompt(orch, project_root)

                user_message = message

                # Build user prompt with conversation context if needed
                if conversation_history:
                    user_prompt = f"""## Previous Conversation
{conversation_history}

---

**Current request:** {user_message}"""
                else:
                    user_prompt = user_message

                # Execute COO via AgentExecutorPool for unified execution
                try:
                    result = await execute_coo_via_pool(
                        websocket=websocket,
                        project_root=project_root,
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                    )
                except asyncio.CancelledError:
                    raise RuntimeError("COO execution was cancelled")
                except Exception as e:
                    logger.error(f"COO execution failed: {e}")
                    raise RuntimeError(f"**COO Execution Error:** {e}")

                # Check if we got a result
                if result is None:
                    raise RuntimeError("**Failed to get response from Claude CLI.**")

                # Send the complete response
                final_content = result["response"]
                if not final_content:
                    final_content = "(No response generated)"

                await manager.send_event(
                    websocket,
                    "agent_complete",
                    {
                        "agent": "Supreme Orchestrator",
                        "agent_type": "orchestrator",
                        "content": final_content,
                        "thinking": result.get("thinking", ""),
                    },
                )

                # Send completion
                await manager.send_event(
                    websocket,
                    "chat_complete",
                    {
                        "success": True,
                    },
                )

                # Save session summary to memory (lightweight, don't block)
                try:
                    session_summary = f"**User**: {message[:200]}{'...' if len(message) > 200 else ''}\n\n**COO Response**: {final_content[:500]}{'...' if len(final_content) > 500 else ''}"
                    memory.save_session_summary(
                        session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                        summary=session_summary,
                        swarm_name=None,  # COO-level, not swarm-specific
                    )
                except Exception as mem_err:
                    logger.warning(f"Failed to save session summary: {mem_err}")

            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                error_msg = str(e)

                # Make error message user-friendly
                if "ANTHROPIC_API_KEY" in error_msg:
                    pass  # Already formatted nicely
                elif "401" in error_msg or "authentication" in error_msg.lower():
                    error_msg = (
                        "**Authentication failed.**\n\n"
                        "Your API key may be invalid.\n\n"
                        "**To fix:**\n"
                        "1. Check your API key at https://console.anthropic.com\n"
                        "2. Update `backend/.env` with: `ANTHROPIC_API_KEY=your_key`\n"
                        "3. Restart the backend server"
                    )

                # Wrap error-path sends in try/except to prevent cascade failures
                try:
                    await manager.send_event(
                        websocket,
                        "agent_complete",
                        {
                            "agent": "Supreme Orchestrator",
                            "agent_type": "orchestrator",
                            "content": error_msg,
                        },
                    )
                    await manager.send_event(
                        websocket,
                        "chat_complete",
                        {
                            "success": False,
                        },
                    )
                except Exception as send_err:
                    logger.debug(f"Failed to send error response (client may have disconnected): {send_err}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
