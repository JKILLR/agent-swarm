"""Tool definitions for agents to use context system.

These are registered with the agent executor and exposed
to Claude via the standard tool interface.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_navigator import ContextNavigator


# =============================================================================
# Core Context Exploration Tools
# =============================================================================

CONTEXT_TOOLS = [
    {
        "name": "context_list",
        "description": """List available contexts. Returns handles (not content).

Use this first to see what contexts exist before exploring.

Args:
    type: Optional filter by type (identity, project, temporal, document, memory)

Returns: List of context handles with metadata (id, name, type, size)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["identity", "project", "temporal", "document", "memory", "tool_result", "web"],
                    "description": "Filter by context type"
                }
            }
        }
    },
    {
        "name": "context_peek",
        "description": """Preview a context without loading all content.

Returns first ~100 tokens to understand structure and relevance.
Use this before deciding whether to load full content.

Args:
    context_id: ID of the context to preview
    lines: Number of lines to preview (default: 10)

Returns: Preview text with truncation info""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": "ID of context to preview"
                },
                "lines": {
                    "type": "integer",
                    "default": 10,
                    "description": "Lines to preview"
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_grep",
        "description": """Search within a context for a pattern.

Finds relevant sections without loading all content.
Returns matches with surrounding context lines.

Args:
    context_id: ID of context to search
    pattern: Regex pattern to search for
    context_lines: Lines of context around matches (default: 2)

Returns: List of matches with line numbers and context""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string",
                    "description": "ID of context to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern"
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2
                }
            },
            "required": ["context_id", "pattern"]
        }
    },
    {
        "name": "context_chunk",
        "description": """Get a specific chunk of a context.

Use for paginating through large contexts.
Each chunk is ~50 lines by default.

Args:
    context_id: ID of context
    chunk_index: Which chunk (0-indexed)
    chunk_size: Lines per chunk (default: 50)

Returns: Chunk content with navigation info""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string"
                },
                "chunk_index": {
                    "type": "integer",
                    "default": 0
                },
                "chunk_size": {
                    "type": "integer",
                    "default": 50
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_load",
        "description": """Load full content of a context.

WARNING: Use sparingly! Prefer peek/grep/chunk for exploration.
Only load full content when you specifically need it all.

Args:
    context_id: ID of context to load

Returns: Full content (may be large)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {
                    "type": "string"
                }
            },
            "required": ["context_id"]
        }
    },
    {
        "name": "context_search",
        "description": """Search across ALL contexts for a query.

Finds which contexts contain relevant information.
Use to discover where information lives.

Args:
    query: Search query

Returns: List of contexts with match counts""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
]

# =============================================================================
# Session State Tools (RLM Gap Fix: Cross-Call State Persistence)
# Enables Strategy C: Answer Verification - reference previous search results
# =============================================================================

STATE_TOOLS = [
    {
        "name": "state_set",
        "description": """Store a value in session state for later retrieval.

Use to persist data across tool calls, enabling:
- Store search results for later verification
- Save candidate lists for refinement
- Keep track of findings during exploration

Args:
    key: Name for this state entry (e.g., "candidates", "search_results")
    value: Any JSON-serializable value (string, number, list, dict)

Returns: Confirmation with list of all stored keys""",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "State key name"
                },
                "value": {
                    "description": "Value to store (any JSON type)"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "state_get",
        "description": """Retrieve a value from session state.

Args:
    key: State key to retrieve

Returns: Stored value, or indication that key doesn't exist""",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "State key to retrieve"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "state_list",
        "description": """List all keys in session state with previews.

Returns: All stored keys with value type/size hints""",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "state_clear",
        "description": """Clear all session state.

Use to reset state when starting a new exploration task.

Returns: Confirmation with count of items cleared""",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# =============================================================================
# Result Buffer Tools (RLM Gap Fix: Output Accumulation)
# Enables Strategy D: Long Output Assembly from RLM paper
# =============================================================================

BUFFER_TOOLS = [
    {
        "name": "buffer_append",
        "description": """Append content to the result accumulation buffer.

Use to build complex answers iteratively:
- Process chunks and append summaries
- Collect findings across multiple contexts
- Assemble long outputs piece by piece

Args:
    content: Content to append to buffer
    label: Optional label for this entry (helps track what each entry is)

Returns: Buffer status (size, total chars)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to append"
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for reference"
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "buffer_read",
        "description": """Read all accumulated buffer contents.

Returns all appended content joined together.

Args:
    join_separator: Separator between entries (default: double newline)

Returns: Combined buffer contents with metadata""",
        "input_schema": {
            "type": "object",
            "properties": {
                "join_separator": {
                    "type": "string",
                    "default": "\n\n",
                    "description": "Separator between entries"
                }
            }
        }
    },
    {
        "name": "buffer_clear",
        "description": """Clear the result buffer.

Use after reading buffer to reset for next task.

Returns: Confirmation with count of entries cleared""",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "buffer_pop",
        "description": """Remove and return an entry from the buffer.

Args:
    index: Index to pop (default: -1 for last entry)

Returns: Popped content and remaining buffer size""",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "default": -1,
                    "description": "Index to pop (-1 for last)"
                }
            }
        }
    }
]


# =============================================================================
# Batch Operation Tools (RLM Gap Fix: Parallel Processing)
# Enables faster exploration of large context sets
# =============================================================================

BATCH_TOOLS = [
    {
        "name": "context_batch_grep",
        "description": """Search multiple contexts in parallel.

Enables faster exploration by executing grep operations concurrently.
Use when you need to search across many contexts at once.

Args:
    queries: List of {context_id, pattern} objects to search
    context_lines: Lines of context around matches (default: 2, applied to all)

Returns: List of results, one per query, with matches and totals""",
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "context_id": {
                                "type": "string",
                                "description": "ID of context to search"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern to search for"
                            }
                        },
                        "required": ["context_id", "pattern"]
                    },
                    "description": "List of search queries"
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2,
                    "description": "Lines of context around matches"
                }
            },
            "required": ["queries"]
        }
    },
    {
        "name": "context_batch_peek",
        "description": """Preview multiple contexts in parallel.

Enables faster exploration by fetching previews concurrently.
Use when you need to quickly survey many contexts at once.

Args:
    context_ids: List of context IDs to preview
    chars: Character limit for each preview (default: 400)

Returns: List of previews, one per context, with metadata""",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of context IDs to preview"
                },
                "chars": {
                    "type": "integer",
                    "default": 400,
                    "description": "Character limit for each preview"
                }
            },
            "required": ["context_ids"]
        }
    }
]


def handle_context_tool(
    tool_name: str,
    tool_input: dict,
    navigator: "ContextNavigator"
) -> dict:
    """Route tool calls to navigator methods.

    Args:
        tool_name: Name of the context tool being called
        tool_input: Input parameters for the tool
        navigator: ContextNavigator instance to use

    Returns:
        Result from the navigator method
    """
    # Core context exploration handlers
    context_handlers = {
        "context_list": lambda: navigator.list_contexts(tool_input.get("type")),
        "context_peek": lambda: navigator.peek(
            tool_input["context_id"],
            tool_input.get("lines", 10)
        ),
        "context_grep": lambda: navigator.grep(
            tool_input["context_id"],
            tool_input["pattern"],
            tool_input.get("context_lines", 2)
        ),
        "context_chunk": lambda: navigator.chunk(
            tool_input["context_id"],
            tool_input.get("chunk_index", 0),
            tool_input.get("chunk_size", 50)
        ),
        "context_load": lambda: navigator.load(tool_input["context_id"]),
        "context_search": lambda: navigator.search_all(tool_input["query"])
    }

    # Session state handlers (RLM gap fix)
    state_handlers = {
        "state_set": lambda: navigator.set_state(
            tool_input["key"],
            tool_input["value"]
        ),
        "state_get": lambda: navigator.get_state(tool_input["key"]),
        "state_list": lambda: navigator.list_state(),
        "state_clear": lambda: navigator.clear_state()
    }

    # Buffer handlers (RLM gap fix)
    buffer_handlers = {
        "buffer_append": lambda: navigator.buffer_append(
            tool_input["content"],
            tool_input.get("label")
        ),
        "buffer_read": lambda: navigator.buffer_read(
            tool_input.get("join_separator", "\n\n")
        ),
        "buffer_clear": lambda: navigator.buffer_clear(),
        "buffer_pop": lambda: navigator.buffer_pop(
            tool_input.get("index", -1)
        )
    }

    # Batch operation handlers (RLM gap fix)
    batch_handlers = {
        "context_batch_grep": lambda: navigator.batch_grep(
            tool_input["queries"],
            tool_input.get("context_lines", 2)
        ),
        "context_batch_peek": lambda: navigator.batch_peek(
            tool_input["context_ids"],
            tool_input.get("chars", 400)
        )
    }

    # Merge all handlers
    handlers = {**context_handlers, **state_handlers, **buffer_handlers, **batch_handlers}

    handler = handlers.get(tool_name)
    if handler:
        try:
            return handler()
        except KeyError as e:
            return {"error": f"Missing required parameter: {e}"}
        except Exception as e:
            return {"error": f"Tool execution error: {e}"}
    return {"error": f"Unknown tool: {tool_name}"}


def get_context_tools() -> list[dict]:
    """Get all context tool definitions (exploration only).

    Returns:
        List of tool definition dictionaries
    """
    return CONTEXT_TOOLS.copy()


def get_state_tools() -> list[dict]:
    """Get session state tool definitions.

    Returns:
        List of state tool definitions
    """
    return STATE_TOOLS.copy()


def get_buffer_tools() -> list[dict]:
    """Get result buffer tool definitions.

    Returns:
        List of buffer tool definitions
    """
    return BUFFER_TOOLS.copy()


def get_batch_tools() -> list[dict]:
    """Get batch operation tool definitions.

    Returns:
        List of batch tool definitions
    """
    return BATCH_TOOLS.copy()


def get_all_context_tools() -> list[dict]:
    """Get ALL context-related tools (exploration + state + buffer + batch).

    Use this to get the complete set of RLM-inspired tools.

    Returns:
        List of all tool definitions
    """
    return CONTEXT_TOOLS + STATE_TOOLS + BUFFER_TOOLS + BATCH_TOOLS


def is_context_tool(tool_name: str) -> bool:
    """Check if a tool name is a context-related tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if this is a context, state, or buffer tool
    """
    context_prefixes = ("context_", "state_", "buffer_")
    return tool_name.startswith(context_prefixes)
