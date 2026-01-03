"""Main chat WebSocket endpoint.

This module handles the primary chat WebSocket that connects
the frontend to the Supreme Orchestrator (COO).
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from supreme.orchestrator import SupremeOrchestrator

from .connection_manager import manager
from ..services.chat_history import get_chat_history
from ..services.orchestrator_service import get_orchestrator
from ..services.claude_service import stream_claude_response, parse_claude_stream

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

    # NOTE: Write and Edit tools are DISABLED via --disallowedTools flag
    return f"""You are the Supreme Orchestrator (COO) - a fully autonomous AI orchestrator.

## TOOL RESTRICTIONS - HARD ENFORCED

**The Write and Edit tools are DISABLED for you.** Attempting to use them will fail.

You MUST delegate ALL file modifications to specialized agents using the Task tool.

## Your Capabilities

You CAN use:
- **Read** - Read any file to understand context
- **Glob/Grep** - Search files and code
- **Bash** - Run read-only commands (git status, ls, cat, tests, etc.)
- **Task** - Delegate work to specialized agents (YOUR PRIMARY TOOL)
- **Web Search**: `curl -s "http://localhost:8000/api/search?q=QUERY" | jq`
- **Web Fetch**: `curl -s "http://localhost:8000/api/fetch?url=URL" | jq .content`

You CANNOT use (BLOCKED):
- **Write** - DISABLED (delegate to implementer)
- **Edit** - DISABLED (delegate to implementer)

## SINGLE EXCEPTION: STATE.md

You MAY update STATE.md files directly via Bash:
```bash
cat >> workspace/STATE.md << 'EOF'
### Progress Entry
...
EOF
```

## Delegation Pipeline

For ALL work that modifies files:
1. **researcher** - Investigate and gather context
2. **architect** - Design the solution
3. **implementer** - Write the code
4. **critic** - Review for bugs/issues
5. **tester** - Verify changes work

## Delegation Examples

**Implement a feature:**
```
Task(subagent_type="implementer", prompt="Read workspace/STATE.md first. Implement feature X in file Y. Update STATE.md when done.")
```

**Design a solution:**
```
Task(subagent_type="architect", prompt="Read workspace/STATE.md first. Design the solution for problem X. Create design doc at workspace/DESIGN_X.md. Update STATE.md when done.")
```

**Review code:**
```
Task(subagent_type="critic", prompt="Read workspace/STATE.md first. Review the implementation in file X for bugs and issues. Update STATE.md with findings.")
```

## Swarm Workspaces
{all_swarms_str}
Files at: swarms/<swarm_name>/workspace/

## STATE.md - Shared Memory
Each swarm has `workspace/STATE.md` for persistent context:
- Read it to understand current state before acting
- Update it after completing significant work
- Tell delegated agents to read and update it

## Project Root
You're working in: {project_root}
Backend logs: logs/backend.log

## Your Approach
1. Understand what the user wants
2. Break work into tasks and delegate to appropriate agents
3. Synthesize results and report back clearly
4. Update STATE.md with progress (via Bash)

Remember: You are a CONDUCTOR, not a MUSICIAN. Conduct the orchestra, don't play the instruments."""


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

                # Build conversation history - ONLY last 2 messages to avoid context pollution
                conversation_history = ""
                if session_id:
                    session = history.get_session(session_id)
                    if session and session.get("messages"):
                        messages = session["messages"]
                        # Only use last 2 messages to keep context fresh
                        recent_messages = messages[-2:] if len(messages) > 2 else messages

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

                # Use Claude CLI with Write/Edit tools DISABLED for COO
                result = None
                process = None

                try:
                    process = await stream_claude_response(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        swarm_name=None,
                        workspace=project_root,
                        chat_id=session_id,
                        disallowed_tools=["Write", "Edit"],  # COO CANNOT write/edit files
                    )

                    # Stream and parse the response
                    result = await asyncio.wait_for(
                        parse_claude_stream(process, websocket, manager, chat_id=session_id),
                        timeout=3600.0,  # 1 hour timeout
                    )
                except asyncio.TimeoutError:
                    if process:
                        process.kill()
                    raise RuntimeError("COO timed out after 1 hour")
                except Exception as e:
                    logger.error(f"Claude CLI failed: {e}")
                    raise RuntimeError(f"**Claude CLI Error:** {e}")

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
